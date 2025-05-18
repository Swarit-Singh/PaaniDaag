import json

def metadata_to_bits(metadata_dict):
    """Convert metadata dictionary to a list of bits."""
    json_str = json.dumps(metadata_dict)
    byte_data = json_str.encode('utf-8')
    bits = []
    for byte in byte_data:
        bits.extend([int(b) for b in format(byte, '08b')])
    return bits

def bits_to_metadata(bits):
    """Convert list of bits back to metadata dictionary."""
    if len(bits) % 8 != 0:
        raise ValueError("Bit length must be a multiple of 8.")
    byte_list = [int(''.join(map(str, bits[i:i+8])), 2) for i in range(0, len(bits), 8)]
    json_str = bytes(byte_list).decode('utf-8')
    return json.loads(json_str)

def get_metadata_bit_length(metadata_dict):
    """Returns the number of bits needed to encode the metadata."""
    return len(metadata_to_bits(metadata_dict))
