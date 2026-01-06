import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    LogitsProcessorList,
    LogitsProcessor
)
import math

class TemperatureWarper(LogitsProcessor):
    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        return scores / self.temperature

class EESWarper(LogitsProcessor):
    def __init__(self, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        probs = torch.softmax(scores, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        # probs_mass = torch.cumsum(sorted_probs, dim=-1)
        
        # # Create a mask to mark the token that needs to be removed
        # indices_to_remove = torch.zeros_like(sorted_probs, dtype=torch.bool)

        # for batch_idx in range(sorted_probs.shape[0]):
        #     # Obtain the valid probability (non-zero probability) of the current batch
        #     valid_probs = sorted_probs[batch_idx][sorted_probs[batch_idx] > 0]

        #     if len(valid_probs) > self.min_tokens_to_keep:  # At least two candidate tokens are required for screening
        #         i=self.min_tokens_to_keep+1
        #         for _ in range(self.min_tokens_to_keep+1, 1000):
        #             entropy = -torch.sum(valid_probs[:i]/probs_mass[batch_idx][i-1] * torch.log(valid_probs[:i]/probs_mass[batch_idx][i-1] + 1e-10))
        #             entropy_normalize = entropy/torch.log(torch.tensor(i))
        #             if entropy_normalize < probs_mass[batch_idx][i-1]:
        #                 break
        #             i += 1
        #         # Mark the positions that need to be removed
        #         indices_to_remove[batch_idx, i-1:] = True
        
        # Vectorized process all batches
        batch_size, vocab_size = sorted_probs.shape

        for batch_idx in range(batch_size):
            # Obtain the valid probability (non-zero probability) of the current batch
            valid_probs = sorted_probs[batch_idx][sorted_probs[batch_idx] > 0]

            if len(valid_probs) > self.min_tokens_to_keep:  # At least two candidate tokens are required for screening
                # The state of min_tokens_to_keep tokens before initialization
                k = self.min_tokens_to_keep
                P_k = probs_mass[batch_idx][k-1]

                # Calculate the initial entropy H_k
                normalized_probs_k = valid_probs[:k] / P_k
                H_k = -torch.sum(normalized_probs_k * torch.log(normalized_probs_k + 1e-10))

                # Increment calculation starts from the min_tokens_to_keep+1 token
                for i in range(k+1, len(valid_probs)+1):
                    # Current parameter
                    p_new = valid_probs[i-1]  # The newly added probability
                    P_k_plus_1 = probs_mass[batch_idx][i-1]  # New cumulative probability

                    # Calculate H_{k+1} using the incremental formula
                    ratio = P_k / P_k_plus_1

                    term1 = ratio * H_k
                    term2 = ratio * torch.log(P_k_plus_1 / P_k + 1e-10)
                    term3 = -(p_new / P_k_plus_1) * torch.log(p_new / P_k_plus_1 + 1e-10)

                    H_k_plus_1 = term1 + term2 + term3

                    # Calculate the normalized entropy
                    entropy_normalize = H_k_plus_1 / torch.log(torch.tensor(float(i)))

                    # Check the truncation conditions
                    if entropy_normalize < P_k_plus_1:
                        # Mark the positions that need to be removed (note the index correspondence)
                        indices_to_remove[batch_idx, i-1:] = True
                        break
                    
                    # Update the status for the next iteration
                    H_k = H_k_plus_1
                    P_k = P_k_plus_1

        # Restore the sorted masks to their original order
        original_indices_to_remove = torch.zeros_like(probs, dtype=torch.bool)
        original_indices_to_remove.scatter_(-1, sorted_indices, indices_to_remove)

        # Apply the mask to the original fraction
        scores_processed = scores.masked_fill(original_indices_to_remove, self.filter_value)

        return scores_processed
    

class MirostatWarper(LogitsProcessor):
    
    def __init__(
        self, 
        mirostat_tau: float = 5.0,  # Target cross-entropy τ
        mirostat_eta: float = 0.1,  # Learning rate η
        filter_value: float = -float("Inf"),
        min_tokens_to_keep: int = 1,
        m: int = 100  # Used to estimate the sample size of s
    ):
        if mirostat_tau <= 0:
            raise ValueError(f"mirostat_tau must be positive, got {mirostat_tau}")
        if mirostat_eta <= 0:
            raise ValueError(f"mirostat_eta must be positive, got {mirostat_eta}")
            
        self.tau = mirostat_tau  # Target cross-entropy τ
        self.eta = mirostat_eta  # Learning rate η
        self.mu = None  # The maximum cross-entropy μ = 2τ, with the length of batch_size
        self.m = m  # Sample size
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

        self.last_probs = None  # Record the probability distribution obtained from the last sampling

    def estimate_s(self, prob):
        """
        Calculate sˆ according to formula: sˆ = Σ(tibi) / Σ(ti²)
        
        Args:
            prob: The probability distribution arranged in descending order
            
        Returns:
            float: Estimated shape parameter sˆ
        """
        numerator = 0.0  # Σ(tibi)
        denominator = 0.0  # Σ(ti²)
        epsilon = 1e-9
        
        # Use m-1 samples (i from 1 to m-1)
        max_samples = min(self.m - 1, len(prob) - 1)
        
        for i in range(1, max_samples + 1):  # i starts from 1
            if i >= len(prob):
                break
                
            # bi = pi/pi+1 (Probability ratio)
            bi = prob[i-1] / (prob[i] + epsilon)  # Note the index: i-1 corresponds to pi, and i corresponds to pi+1
            
            # ti = log((i+1)/i)
            ti = math.log((i + 1) / i)
            
            numerator += ti * bi
            denominator += ti * ti
        
        if denominator == 0:
            return 1.0  # Default value
            
        s_hat = numerator / denominator
        return max(s_hat, 1.001)  # Make sure s > 1

    def compute_k(self, N, s_hat, mu):
        """
        Calculate k according to formula: k = (ε̂2^μ/(1-N^(-ε̂)))^(1/ŝ)
        Where ε̂ = ŝ - 1
        
        Args:
            N: Vocabulary size
            s_hat: Estimated shape parameter ŝ
            mu: Current maximum cross-entropy μ
            
        Returns:
            int: The calculated value of k
        """
        try:
            eps_hat = s_hat - 1  # ε̂ = ŝ - 1
            
            if eps_hat <= 0:
                return min(int(2 * self.tau), N - 1)
            
            # Calculate k = (ε̂2^μ/(1-N^(-ε̂)))^(1/ŝ)
            numerator = eps_hat * (2 ** mu)
            denominator = 1 - (N ** (-eps_hat))
            
            if denominator <= 0:
                return min(int(2 * self.tau), N - 1)
                
            k = (numerator / denominator) ** (1 / s_hat)
            k = max(1, min(round(k), N - 1))
            return int(k)
            
        except Exception:
            # If calculation fails, return conservative k value
            return min(int(2 * self.tau), N - 1)

    def compute_cross_entropy(self, probs, selected_token_id):
        """
        Calculate the cross-entropy S(X) of the selected token
        
        Args:
            probs: Probability distribution
            selected_token_id: Selected token ID
            
        Returns:
            float: Cross-entropy value
        """
        # S(X) = -log2(P(X))
        prob_x = probs[selected_token_id].item()
        if prob_x <= 0:
            return float('inf')
        return -math.log2(prob_x)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        batch_size, vocab_size = scores.shape
        if self.last_probs is not None:
            self.update_mu_after_sampling(self.last_probs,input_ids[:,-1])
        
        if self.mu is None:
            self.mu = [2 * self.tau] * batch_size

        processed_scores = scores.clone()
        
        for batch_idx in range(batch_size):
            batch_scores = scores[batch_idx]
            
            # Step 1: Convert to probability distribution and sort
            probs = F.softmax(batch_scores, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            sorted_probs_cpu = sorted_probs.cpu().numpy()
            
            # Step 2: Calculate sˆ (according to formula 30)
            s_hat = self.estimate_s(sorted_probs_cpu)
            
            # Step 3: Calculate k (according to formula 2)
            k = self.compute_k(vocab_size, s_hat, self.mu[batch_idx])
            
            # Step 4: Apply top-k filtering
            if k < vocab_size:
                # Get the k-th largest score as threshold
                top_k_scores, _ = torch.topk(batch_scores, k)
                threshold = top_k_scores[-1]
                
                # Filter tokens below threshold
                indices_to_remove = batch_scores < threshold
                processed_scores[batch_idx] = batch_scores.masked_fill(
                    indices_to_remove, self.filter_value
                )
        
        # Save current probability distribution for next update
        self.last_probs = F.softmax(processed_scores, dim=-1)

        return processed_scores
    
    def update_mu_after_sampling(self, probs, selected_token_id):
        """
        Update μ value after sampling (this method needs to be called after sampling)
        
        Args:
            probs: Probability distribution during sampling
            selected_token_id: Actually selected token ID
        """
        # Step 5: Calculate error e = S(X) - τ
        batch_size = probs.shape[0]
        for batch_idx in range(batch_size):
            cross_entropy = self.compute_cross_entropy(probs[batch_idx], selected_token_id[batch_idx])
            error = cross_entropy - self.tau

            # Step 6: Update μ: μ = μ - ηe
            self.mu[batch_idx] = self.mu[batch_idx] - self.eta * error

            # Ensure μ doesn't become negative or too small
            self.mu[batch_idx] = max(self.mu[batch_idx], 0.1)


class AdaptiveWarper(LogitsProcessor):
    def __init__(
        self, 
        epsilon: float = 0.1,
        filter_value: float = -float("Inf"),
        min_tokens_to_keep: int = 1
    ):
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
            
        self.epsilon = epsilon
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def compute_delta_conf(self, sorted_probs: torch.Tensor, vocab_size: int) -> torch.Tensor:
        """
        Calculate ΔConf values
        
        Args:
            sorted_probs: Sorted probability distribution [vocab_size]
            vocab_size: Vocabulary size
            
        Returns:
            delta_conf: ΔConf values [vocab_size]
        """
        device = sorted_probs.device
        
        # Calculate cumulative sum
        cumsum = torch.cumsum(sorted_probs, dim=0)  # [vocab_size]
        
        # Calculate residual index = |V| - range(1, |V| + 1)
        # range(1, |V| + 1) = [1, 2, 3, ..., |V|]
        # residual_index = [|V|-1, |V|-2, |V|-3, ..., 0]
        range_indices = torch.arange(1, vocab_size + 1, device=device, dtype=torch.float)
        residual_index = vocab_size - range_indices  # [vocab_size]
        
        # Avoid division by zero and log zero issues
        eps = 1e-10
        
        # Calculate term1 = p * log(p * residual_index / (1 - cumsum))
        # Note: cumsum is the cumulative sum up to current position, so 1-cumsum is remaining probability
        remaining_prob = 1 - cumsum  # Remaining probability including current token
        remaining_prob = torch.clamp(remaining_prob, min=eps)
        residual_index = torch.clamp(residual_index, min=eps)
        
        numerator1 = sorted_probs * residual_index
        numerator1 = torch.clamp(numerator1, min=eps)
        
        term1 = sorted_probs * torch.log(numerator1 / remaining_prob)
        
        # Calculate term2 = log((1-cumsum)/residual_index) - log((1-cumsum+p)/(residual_index+1))
        
        residual_index_plus1 = residual_index + 1
        residual_index_plus1 = torch.clamp(residual_index_plus1, min=eps)
        
        log_term1 = torch.log(remaining_prob / residual_index)
        log_term2 = torch.log(remaining_prob + sorted_probs / residual_index_plus1)
        
        term2 = log_term1 - log_term2
        
        # Calculate final ΔConf
        # ΔConf = (term1 + (1-cumsum+p) * term2) / log|V|
        delta_conf = (term1 + (remaining_prob + sorted_probs) * term2) / torch.log(torch.tensor(vocab_size,device=device))
        
        return delta_conf

    def find_cutoff_k(self, delta_conf: torch.Tensor) -> int:
        """
        Find k value that satisfies the condition
        k = max(1, LastIndex(ΔConf > ε))
        
        Args:
            delta_conf: ΔConf values [vocab_size]
            
        Returns:
            k: Cutoff position
        """
        # Find all positions where ΔConf > ε
        valid_indices = torch.where(delta_conf > self.epsilon)[0]
        
        if len(valid_indices) == 0:
            # If none satisfy the condition, return minimum value
            return max(1, self.min_tokens_to_keep)
        
        # Return the last index that satisfies the condition +1 (since index starts from 0)
        k = valid_indices[-1].item() + 1
        return max(k, self.min_tokens_to_keep)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Apply Adaptive Decoding
        
        Args:
            input_ids: Input token sequence [batch_size, seq_len]
            scores: Logits scores [batch_size, vocab_size]
            
        Returns:
            processed_scores: Processed logits [batch_size, vocab_size]
        """
        batch_size, vocab_size = scores.shape
        processed_scores = scores.clone()
        
        # Process each batch independently
        for batch_idx in range(batch_size):
            batch_scores = scores[batch_idx]  # [vocab_size]
            
            # Convert to probability distribution and sort
            probs = F.softmax(batch_scores, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            
            # Calculate ΔConf
            try:
                delta_conf = self.compute_delta_conf(sorted_probs, vocab_size)
                
                # Find cutoff position k
                k = self.find_cutoff_k(delta_conf)
                
                # Apply top-k filtering
                if k < vocab_size:
                    # Get top-k threshold
                    kth_score = torch.topk(batch_scores, k)[0][-1]
                    
                    # Filter out tokens not in top-k
                    indices_to_remove = batch_scores < kth_score
                    processed_scores[batch_idx] = batch_scores.masked_fill(
                        indices_to_remove, self.filter_value
                    )
                    
            except Exception as e:
                # If error occurs during calculation, keep original scores
                print(f"Warning: Adaptive decoding failed for batch {batch_idx}: {e}")
                continue
        
        return processed_scores
