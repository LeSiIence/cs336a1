# cs336 assignment 1

## Sec.2 BPE Tokenizer

### 2.1

- (a) What Unicode character does chr(0) return? 

  a non-printable control character whose official name is `NULL`.

- (b)How does this character’s string representation (`__repr__()`) differ from its printed representation?

  `\x00` and ` `

- (c)What happens when this character occurs in text? It may be helpful to play around with the following in your Python interpreter and see if it matches your expectations:

  (c) When U+0000 occurs in text, it remains part of the string but is normally invisible when printed(except for some frontend rendering), so the surrounding visible characters appear adjacent.

  ```bash
  'this is a test\x00string'
  >>> chr(0)
  '\x00'
  >>> print(chr(0))
  
  >>> "this is a test" + chr(0) + "string"
  'this is a test\x00string'
  >>> print("this is a test" + chr(0) + "string")
  this is a teststring
  ```

### 2.2

- (a)What are some reasons to prefer training our tokenizer on UTF-8 encoded bytes, rather than UTF-16 or UTF-32? It may be helpful to compare the output of these encodings for various input strings.

  Because most of the training text, which is 98%+ of the web pages are encoded in utf-8. Besides, using utf-8 is more compact for text only in ascii characters.

- (b)Consider the following (incorrect) function, which is intended to decode a UTF-8 byte string into a Unicode string. Why is this function incorrect? Provide an example of an input byte string that yields incorrect results.

  ```python
  def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
  return "".join([bytes([b]).decode("utf-8") for b in bytestring])
  ```

  

  utf-8 bytestring cannot be interpreted byte by byte for there are characters taking more than one byte. in the case below, those string with characters of more than one byte, will never be interpreted correctly. For example, `"hello! こんにちは!"`

- (c)Give a two-byte sequence that does not decode to any Unicode character(s).

  `b'\xff\xff'`, for the start of 1-byte char is `0xxxxxxx`, for 2-bytes:`110xxxxx`, for 3-bytes:`1110xxxx` and for 4-bytes:`11110xxx` and the following bytes must be `10xxxxxx`

### 2.3

















