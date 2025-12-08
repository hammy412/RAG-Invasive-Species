import { useState, useRef, useEffect } from "react";
import "./ChatbotUI.css";

interface Message {
  role: "assistant" | "user";
  content: string;
}

export default function ChatbotUI() {
  const [messages, setMessages] = useState<Message[]>([
    { role: "assistant", content: "Hello! How can I help you today?" },
  ]);
  const [input, setInput] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(false);
  const messagesEndRef = useRef<HTMLDivElement | null>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(scrollToBottom, [messages, loading]);

  const sendMessage = async () => {
    if (!input.trim()) return;

    const userMessage: Message = { role: "user", content: input };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setLoading(true);

    try {
      // POST to your local backend
      const res = await fetch("http://127.0.0.1:8000/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: input }), // send { "question": "..." }
      });

      if (!res.ok) throw new Error("Network response was not ok");

      const data = await res.json();

      const botMessage: Message = {
        role: "assistant",
        content: data.answer || "Sorry, no response from the API.",
      };

      setMessages((prev) => [...prev, botMessage]);
    } catch (error) {
      console.error(error);
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: "Error contacting the API." },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="chat-container">
      <div className="chat-window">
        {messages.map((msg, idx) => (
          <div
            key={idx}
            className={`chat-message ${msg.role === "assistant" ? "assistant" : "user"}`}
          >
            {msg.content}
          </div>
        ))}
        {loading && <div className="chat-message assistant">Thinking...</div>}
        <div ref={messagesEndRef} />
      </div>

      <div className="chat-input-container">
        <input
          className="chat-input"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Type a message..."
          onKeyDown={(e) => e.key === "Enter" && sendMessage()}
        />
        <button className="chat-send-button" onClick={sendMessage} disabled={loading}>
          Send
        </button>
      </div>
    </div>
  );
}
