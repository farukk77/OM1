import logging
from typing import Optional, Any
# Importing eth_abi for strict decoding
from eth_abi import decode
from eth_utils import to_bytes
from eth_abi.exceptions import DecodingError

logger = logging.getLogger(__name__)

# ... (Leave the rest of the class untouched, only replace the function) ...

    def decode_eth_response(self, hex_response: Optional[str]) -> str:
        """
        Decodes Ethereum RPC response using eth-abi for strict validation.
        Fixes #1824 by distinguishing between valid empty data and malformed/network errors.
        """
        # 1. SCENARIO B: Network Error / Null Response
        # Handle RPC errors or timeouts returning None.
        if hex_response is None:
            logger.error("RPC returned None. Possible network timeout or node issue.")
            return ""

        # 2. Basic Type Validation
        if not isinstance(hex_response, str):
            logger.warning(f"Invalid hex format received: Expected string, got {type(hex_response)}")
            return ""

        # 3. SCENARIO A: Valid Empty Response
        # If the contract actually returned empty data (0x), this is valid.
        if hex_response == "0x":
            logger.debug("Received valid empty '0x' response.")
            return ""

        try:
            # Safely convert hex string to bytes
            byte_data = to_bytes(hexstr=hex_response)

            # 4. SCENARIO C: Strict Decoding
            # Using eth_abi instead of manual slicing.
            # If data is corrupted or incomplete (Scenario C), this raises 'DecodingError'.
            # This prevents "Silent Failures" and catches the error explicitly.
            decoded_tuple = decode(['string'], byte_data)
            
            # decode always returns a tuple, getting the first element.
            return decoded_tuple[0]

        except DecodingError as e:
            # Critical: Log if the data is corrupted or truncated.
            logger.error(f"ABI Decoding Failed (Corrupted/Truncated Data): {e}. Raw Data: {hex_response}")
            return ""
            
        except Exception as e:
            logger.error(f"Unexpected error in decode_eth_response: {e}")
            return ""