import React from 'react';
import { createRoot } from 'react-dom/client';
import { Provider } from 'react-redux';   // Import Provider from react-redux
import { store } from './redux/store';    // Import the Redux store
import './styles/index.css';
import './styles/tailwind.css';
import App from './App.jsx';

createRoot(document.getElementById('root')).render(
  <Provider store={store}>                {/* Wrap App with Provider */}
    <React.StrictMode>
      <App />
    </React.StrictMode>
  </Provider>
);
