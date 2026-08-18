"""Provider-specific OCR prompts.

The transport contract is shared by the OpenAI-compatible adapters, but the
instruction sent to each model is intentionally explicit and versionable.
"""

LABEL_JSON_SCHEMA = """Return ONLY valid JSON. Do not use Markdown fences and do not add explanations.
Preserve every visible character exactly as shown; do not correct, normalize,
uppercase, or reinterpret OCR characters. Use an empty string only when a
field is not visible.

Use exactly this schema:
{
  "raw_text": "all visible text in reading order",
  "fields": {
    "sku": "",
    "lot": "",
    "quantity": "",
    "unit": "",
    "serial": "",
    "po_number": "",
    "barcode": ""
  }
}

Keep quantity and unit separate. The barcode field is the visible value next
to CODE, QR, or BARCODE; do not invent a value from a decoder.
"""


GLM_OCR_PROMPT = "Text Recognition:\nRead the entire product label from the image.\n\n" + LABEL_JSON_SCHEMA
MONKEY_OCR_PROMPT = "Document parsing:\nParse the complete product label and extract its fields.\n\n" + LABEL_JSON_SCHEMA
QWEN_OCR_PROMPT = "Perform precise OCR on the product label image and return the structured label data.\n\n" + LABEL_JSON_SCHEMA
SURYA_OCR_PROMPT = "Transcribe all visible text on the product label in reading order and extract the labeled values.\n\n" + LABEL_JSON_SCHEMA
DEEPSEEK_OCR_PROMPT = "Free OCR followed by label-field extraction. Read the complete image without guessing.\n\n" + LABEL_JSON_SCHEMA
MISTRAL_OCR_PROMPT = "Extract the document text and product-label fields from this image.\n\n" + LABEL_JSON_SCHEMA

