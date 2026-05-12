const OPENAI_API_URL = "https://api.openai.com/v1/responses";

function json(statusCode, body) {
  return {
    statusCode,
    headers: {
      "Content-Type": "application/json; charset=utf-8",
      "Cache-Control": "no-store",
    },
    body: JSON.stringify(body),
  };
}

function compactMessages(messages) {
  if (!Array.isArray(messages)) return [];
  return messages
    .slice(-10)
    .map((message) => ({
      role: message.role === "assistant" ? "assistant" : "user",
      content: String(message.content || "").slice(0, 900),
    }))
    .filter((message) => message.content.trim());
}

exports.handler = async (event) => {
  if (event.httpMethod !== "POST") {
    return json(405, { error: "POST requests only" });
  }

  if (!process.env.OPENAI_API_KEY) {
    return json(500, {
      error: "OPENAI_API_KEY is not configured on the server.",
    });
  }

  let payload;
  try {
    payload = JSON.parse(event.body || "{}");
  } catch {
    return json(400, { error: "Invalid JSON body" });
  }

  const system = String(payload.system || "").slice(0, 5000);
  const messages = compactMessages(payload.messages);

  if (!system || messages.length === 0) {
    return json(400, { error: "Missing system prompt or messages" });
  }

  try {
    const response = await fetch(OPENAI_API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
      },
      body: JSON.stringify({
        model: process.env.OPENAI_MODEL || "gpt-5.2",
        instructions: system,
        input: messages,
        max_output_tokens: 420,
      }),
    });

    const data = await response.json();

    if (!response.ok) {
      return json(response.status, {
        error: data?.error?.message || "OpenAI request failed",
      });
    }

    return json(200, {
      reply: data.output_text || "지금은 말이 잘 이어지지 않는구나. 다시 물어봐 주겠니?",
    });
  } catch (error) {
    return json(502, {
      error: "Unable to reach OpenAI from the server function.",
    });
  }
};
