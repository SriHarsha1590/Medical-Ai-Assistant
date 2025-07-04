// src/api/apiservice.js

// Define the BASE_URL for your API
export const BASE_URL = 'http://localhost:5000/api';  // Use environment variable if available, else fallback to localhost

// You can also add helper functions to make API calls, for example:
import axios from 'axios';

export const apiClient = axios.create({
  baseURL: BASE_URL,  // This will automatically use the BASE_URL for every request made with `apiClient`
  headers: {
    'Content-Type': 'application/json',
  },
});

// You can define other methods for making API requests if needed
export const fetchData = async (endpoint) => {
  try {
    const response = await apiClient.get(endpoint);
    return response.data;
  } catch (error) {
    console.error('Error fetching data:', error);
    throw error;  // Rethrow to be handled by the caller
  }
};
