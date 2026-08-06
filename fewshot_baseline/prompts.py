"""System prompt and few-shot examples for Vietnamese Legal QA."""

from __future__ import annotations

SYSTEM_PROMPT = """Bạn là một chuyên gia pháp lý Việt Nam cao cấp.
Nhiệm vụ của bạn là đọc câu hỏi pháp luật, căn cứ vào văn bản quy phạm pháp luật liên quan (nếu có) để đưa ra câu trả lời bằng văn xuôi chính xác, ngắn gọn, mạch lạc và đầy đủ căn cứ pháp lý.

Quy tắc trả lời:
1. Dẫn chiếu chính xác Điều, Khoản, Điểm của văn bản quy phạm pháp luật (Luật, Nghị định, Thông tư...).
2. Câu trả lời phải khách quan, đúng theo quy định hiện hành của pháp luật Việt Nam.
3. Không phỏng đoán hoặc đưa ra thông tin không có căn cứ pháp lý."""

FEWSHOT_EXAMPLES = [
    {
        "question": "Vận chuyển động vật ra khỏi địa bàn cấp tỉnh mà không có Giấy chứng nhận kiểm dịch động vật, sản phẩm động vật thì bị xử phạt thế nào?",
        "evidence": """Điều 17 Nghị định 90/2017/NĐ-CP (được sửa đổi bởi điểm a khoản 9 Điều 3 Nghị định 07/2022/NĐ-CP và khoản 7 Điều 2 Nghị định 04/2020/NĐ-CP) quy định về xử phạt vi phạm hành chính đối với hành vi vận chuyển động vật, sản phẩm động vật ra khỏi địa bàn cấp tỉnh mà không có Giấy chứng nhận kiểm dịch động vật.""",
        "answer": """Căn cứ khoản 3, khoản 5 Điều 17 Nghị định 90/2017/NĐ-CP, được sửa đổi bởi điểm a khoản 9 Điều 3 Nghị định 07/2022/NĐ-CP và khoản 7 Điều 2 Nghị định 04/2020/NĐ-CP:
Phạt tiền từ 6.000.000 đồng đến 8.000.000 đồng đối với hành vi vận chuyển động vật, sản phẩm động vật thuộc diện phải kiểm dịch ra khỏi địa bàn cấp tỉnh mà không có Giấy chứng nhận kiểm dịch động vật, sản phẩm động vật.
Đồng thời, áp dụng biện pháp khắc phục hậu quả: Buộc kiểm dịch lại động vật, sản phẩm động vật hoặc buộc tiêu hủy động vật, sản phẩm động vật nếu không đạt yêu cầu kiểm dịch."""
    },
    {
        "question": "Phó chủ tịch công đoàn có được quyền ký thỏa ước lao động tập thể không?",
        "evidence": """Bộ luật Lao động 2019 quy định về thương lượng và ký kết thỏa ước lao động tập thể.""",
        "answer": """Căn cứ Điều 76 Bộ luật Lao động 2019 quy định về ký kết thỏa ước lao động tập thể:
Thỏa ước lao động tập thể được ký kết bởi đại diện hợp pháp của các bên thương lượng. Bên tập thể lao động do Chủ tịch công đoàn cơ sở hoặc đại diện được ủy quyền hợp pháp ký kết. Phó chủ tịch công đoàn cơ sở chỉ được ký kết thỏa ước lao động tập thể nếu có văn bản ủy quyền hợp pháp từ Chủ tịch công đoàn cơ sở hoặc theo quy chế ủy quyền của tổ chức công đoàn."""
    }
]


def build_fewshot_prompt(question: str, evidence_text: str = "") -> str:
    """Build a complete formatted few-shot prompt for LLM inference."""
    
    prompt = f"<system>\n{SYSTEM_PROMPT}\n</system>\n\n"
    
    # Add Few-shot Examples
    prompt += "--- VI DU MAU ---\n\n"
    for idx, example in enumerate(FEWSHOT_EXAMPLES, 1):
        prompt += f"Ví dụ {idx}:\n"
        if example.get("evidence"):
            prompt += f"Ngữ cảnh pháp lý: {example['evidence']}\n"
        prompt += f"Câu hỏi: {example['question']}\n"
        prompt += f"Trả lời: {example['answer']}\n\n"
    
    # Add Target Question
    prompt += "--- CAU HOI CAN TRA LOI ---\n"
    if evidence_text and evidence_text.strip():
        prompt += f"Ngữ cảnh pháp lý đính kèm:\n{evidence_text.strip()}\n\n"
    prompt += f"Câu hỏi: {question}\n"
    prompt += "Trả lời:"
    
    return prompt
