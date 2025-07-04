// src/components/TeamCommunication.jsx
import React, { useState, useEffect } from "react";
import axios from "axios";
import { useAppContext } from "../context/AppContext";

const TeamCommunication = () => {
  const { user } = useAppContext();
  const [messages, setMessages] = useState([]);
  const [newMessage, setNewMessage] = useState("");

  useEffect(() => {
    if (user) {
      axios
        .get(`/api/team-communications/${user.email}`)
        .then((response) => setMessages(response.data))
        .catch((error) => console.error("Error fetching messages:", error));
    }
  }, [user]);

  const handleSubmit = (e) => {
    e.preventDefault();
    axios
      .post(`/api/team-communications/${user.email}`, { message: newMessage })
      .then((response) => {
        setMessages([...messages, response.data]);
        setNewMessage("");
      })
      .catch((error) => console.error("Error sending message:", error));
  };

  return (
    <div>
      <h2 className="text-2xl font-bold text-navy mb-4">Team Communication</h2>
      <div className="bg-gray-light p-4 h-72 overflow-y-auto">
        {messages.map((msg, index) => (
          <div key={index} className="mb-2">
            <p>{msg.message}</p>
          </div>
        ))}
      </div>
      <form onSubmit={handleSubmit} className="space-y-4 mt-4">
        <textarea
          value={newMessage}
          onChange={(e) => setNewMessage(e.target.value)}
          placeholder="Type a message..."
          className="p-2 w-full border rounded"
        />
        <button type="submit" className="bg-teal text-white p-2 rounded">Send</button>
      </form>
    </div>
  );
};

export default TeamCommunication;
