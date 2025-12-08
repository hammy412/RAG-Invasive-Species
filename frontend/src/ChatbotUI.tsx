import { useState, useRef, useEffect } from "react";
import "./ChatbotUI.css";

interface Chunk {
  content: string;
  source?: string;
  chunk_id?: string | null;
}

interface Message {
  role: "assistant" | "user";
  content: string;
  chunks?: Chunk[];
}

export default function ChatbotUI() {
  const [messages, setMessages] = useState<Message[]>([
    { role: "assistant", content: "Hello! How can I help you today?" },
  ]);
  const [input, setInput] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(false);
  const [showChunks, setShowChunks] = useState<boolean>(false);
  const [selectedMessageIdx, setSelectedMessageIdx] = useState<number | null>(null);
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
      const res = await fetch("http://127.0.0.1:8000/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: input }),
      });

      if (!res.ok) throw new Error("Network response was not ok");

      const data = await res.json();
      const answer = typeof data.answer === "string" ? data.answer : "No response from API";
      const chunks: Chunk[] = Array.isArray(data.retrieved_chunks) ? data.retrieved_chunks : [];

      const botMessage: Message = { role: "assistant", content: answer, chunks };
      setMessages((prev) => [...prev, botMessage]);
      setSelectedMessageIdx(messages.length);
      setShowChunks(false); // hide chunks by default until user toggles
    } catch (error) {
      console.error(error);
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: "Error contacting the API." },
      ]);
      setSelectedMessageIdx(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="chat-container-wrapper">
      <div className="chat-container">
        <div className="chat-main">
          {/* Chat Window */}
          <div className="chat-window">
            {messages.map((msg, idx) => (
              <div
                key={idx}
                className={`chat-message ${msg.role === "assistant" ? "assistant" : "user"}`}
                onClick={() => {
                  if (msg.role === "assistant" && msg.chunks && msg.chunks.length > 0) {
                    setSelectedMessageIdx(idx);
                    setShowChunks(true);
                  }
                }}
              >
                {msg.content}
              </div>
            ))}
            {loading && <div className="chat-message assistant">Thinking...</div>}
            <div ref={messagesEndRef} />
          </div>

          {/* Toggleable Chunks Panel */}
          {showChunks && selectedMessageIdx !== null && messages[selectedMessageIdx].chunks && (
            <div className="chat-chunks">
              <div className="chunks-header">
                <h3>Retrieved Chunks</h3>
                <button onClick={() => setShowChunks(false)}>Close</button>
              </div>
              <div className="chunk-container">
                {messages[selectedMessageIdx].chunks!.map((chunk, idx) => (
                  <div key={idx} className="chunk">
                    {chunk.content}
                    {chunk.source && (
                      <div className="chunk-source">Source: {chunk.source}</div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Input Area */}
        <div className="chat-input-container">
          <input
            className="chat-input"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Type a message..."
            onKeyDown={(e) => e.key === "Enter" && sendMessage()}
          />
          <button
            className="chat-send-button"
            onClick={sendMessage}
            disabled={loading}
          >
            Send
          </button>
        </div>
      </div>
    </div>
  );
}
