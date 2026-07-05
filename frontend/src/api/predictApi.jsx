const API_BASE_URL = "http://localhost:8000";

export async function predict(text) {
  const res = await fetch(`${API_BASE_URL}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });

  if (!res.ok) {
    throw new Error(`Request gagal dengan status ${res.status}`);
  }

  return res.json();
}