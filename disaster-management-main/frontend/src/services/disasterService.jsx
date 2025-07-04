// src/services/disasterService.js

import axios from 'axios';

// Use environment variable for API URL, with fallback to localhost in development
const API_URL = 'http://localhost:5000/api';

// Create an Axios instance
const axiosInstance = axios.create({
  baseURL: API_URL,
  timeout: 5000, // Timeout configuration
  headers: {
    'Content-Type': 'application/json',
  },
});

// Function to fetch all disasters
export const fetchAllDisasters = async () => {
  try {
    const response = await axiosInstance.get('/disasters'); // Endpoint for fetching disasters
    if (response.status === 200) {
      return response.data; // Return disaster data
    } else {
      throw new Error(`Unexpected response code: ${response.status}`);
    }
  } catch (error) {
    console.error('Error fetching disasters:', error.message || error);
    throw error; // Propagate error
  }
};
