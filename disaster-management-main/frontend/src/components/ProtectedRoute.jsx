// src/components/ProtectedRoute.jsx
import React from 'react';
import { useSelector } from 'react-redux';
import { Navigate } from 'react-router-dom';

const ProtectedRoute = ({ element, requiredRole }) => {
  const { currentUser, role } = useSelector((state) => state.user);

  if (!currentUser) {
    return <Navigate to="/login" replace />;
  }
  
  if (requiredRole && role !== requiredRole) {
    return <Navigate to="/" replace />;
  }

  return element;
};

export default ProtectedRoute;
