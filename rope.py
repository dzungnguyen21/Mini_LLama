from typing import Tuple
import torch

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Helper function to reshape frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)

def apply_rotary_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query and key tensors. The rotation to each token
    embedding is a function of that token's position in the sequence, head_dim, and theta.
    The input tensors are reshaped as complex numbers to simplify your implementation.

    Args:
        query (torch.Tensor): Query tensor to apply rotary embeddings.
                              Shape: (batch_size, seqlen, n_local_heads, self.head_dim)
        key (torch.Tensor): Key tensor to apply rotary embeddings.
                              Shape: (batch_size, seqlen, n_local_kv_heads, self.head_dim)
        head_dim (int): Dimension of each attention head.
        max_seq_len (int): Maximum sequence length supported by model.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """

    _, seqlen, _, _ = query.shape
    device = query.device

    if seqlen > max_seq_len:
        seqlen = max_seq_len
        query = query[:, :seqlen, :, :]
        key = key[:, :seqlen, :, :]

    # Chuyển đổi query và key thành dạng số phức (biểu diễn dưới dạng cặp giá trị real và imag)
    query_real, query_imag = query.float().reshape(query.shape[:-1] + (-1, 2)).unbind(-1)
    key_real, key_imag = key.float().reshape(key.shape[:-1] + (-1, 2)).unbind(-1)

    # query_real chứa các phần tử tại chỉ số lẻ (q_1, q_3, q_5, ...)
    # query_imag chứa các phần tử tại chỉ số chẵn (q_2, q_4, q_6, ...)

    # Tính toán ma trận sin và cos để xoay vector
    d = head_dim
    Theta = torch.tensor([theta**(-2 * i / d) for i in range(d//2)]).to(device)
    sin_matrix = torch.stack([torch.sin(m * Theta) for m in range(seqlen)]).reshape([1, seqlen, -1, d//2]).to(device)
    cos_matrix = torch.stack([torch.cos(m * Theta) for m in range(seqlen)]).reshape([1, seqlen, -1, d//2]).to(device)
    
    # Thực hiện phép quay vector (theo công thức số phức):
    query_real, query_imag = query_real * cos_matrix - query_imag * sin_matrix, query_real * sin_matrix + query_imag * cos_matrix
    key_real, key_imag = key_real * cos_matrix - key_imag * sin_matrix, key_real * sin_matrix + key_imag * cos_matrix
    
    # Gộp lại thành tensor đầu ra ban đầu
    query_out = torch.stack((query_real, query_imag), dim=-1).reshape(query.shape)
    key_out = torch.stack((key_real, key_imag), dim=-1).reshape(key.shape)
    
    # Return the rotary position embeddings for the query and key tensors
    return query_out, key_out