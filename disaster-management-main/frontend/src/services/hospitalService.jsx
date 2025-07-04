import axios from "axios";

const BASE_URL = "http://localhost:5000/api/hospitals"; // Modify the URL according to your backend

// Function to get all hospitals
export const getAllHospitals = async () => {
    try {
        const response = await axios.get(BASE_URL);
        return response.data; // return the list of hospitals
    } catch (error) {
        console.error("Error fetching hospitals:", error);
        throw error;
    }
};
