import axios from "axios";

const BASE_URL = "http://localhost:5000/api/shelters"; // Modify the URL according to your backend

// Function to get all shelters
export const getAllShelters = async () => {
    try {
        const response = await axios.get(BASE_URL);
        return response.data; // return the list of shelters
    } catch (error) {
        console.error("Error fetching shelters:", error);
        throw error;
    }
};
