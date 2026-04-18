from duo_attn.patch.streaming_attn import block_streaming_attn_func

try:
    from duo_attn.patch.flashinfer_utils import flashinfer
except ImportError:
    flashinfer = None

try:
    import flash_attn  # noqa: F401
    FLASH_ATTN_AVAILABLE: bool = True
except ImportError:
    FLASH_ATTN_AVAILABLE: bool = False

BLOCKSPARSE_AVAILABLE: bool = block_streaming_attn_func is not None


def is_blocksparse_available() -> bool:
    return BLOCKSPARSE_AVAILABLE
