// src/context/AppContext.jsx
import React, { createContext, useState, useContext } from "react";

const AppContext = createContext();

export const AppProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [notifications, setNotifications] = useState([]);
  const [teamMessages, setTeamMessages] = useState([]);

  return (
    <AppContext.Provider
      value={{ user, setUser, notifications, setNotifications, teamMessages, setTeamMessages }}
    >
      {children}
    </AppContext.Provider>
  );
};

export const useAppContext = () => useContext(AppContext);
