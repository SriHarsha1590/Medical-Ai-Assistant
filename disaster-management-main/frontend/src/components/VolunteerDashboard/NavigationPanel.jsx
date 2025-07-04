// src/components/NavigationPanel.jsx
import React from "react";
import { FaHome, FaUser, FaFileAlt, FaComments, FaBell, FaCog } from "react-icons/fa";

const NavigationPanel = ({ currentPage, setCurrentPage }) => {
  const navItems = [
    { name: "Dashboard", icon: <FaHome />, page: "dashboard" },
    { name: "Profile", icon: <FaUser />, page: "profile" },
    { name: "Reports", icon: <FaFileAlt />, page: "reports" },
    { name: "Team", icon: <FaComments />, page: "team" },
    { name: "Notifications", icon: <FaBell />, page: "notifications" },
    { name: "Settings", icon: <FaCog />, page: "settings" },
  ];

  return (
    <nav className="bg-navy text-white w-64 h-screen flex flex-col py-4">
      <h1 className="text-xl font-bold px-6 mb-6">Volunteer Dashboard</h1>
      {navItems.map((item) => (
        <button
          key={item.page}
          onClick={() => setCurrentPage(item.page)}
          className={`flex items-center px-6 py-3 text-left hover:bg-teal hover:text-navy transition ${
            currentPage === item.page ? "bg-teal text-navy" : ""
          }`}
        >
          {item.icon}
          <span className="ml-4">{item.name}</span>
        </button>
      ))}
    </nav>
  );
};

export default NavigationPanel;
