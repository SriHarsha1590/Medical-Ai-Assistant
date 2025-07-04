import React, { useState, useEffect } from "react";
import axios from "axios";
import { GoogleMap, Marker, InfoWindow, useJsApiLoader } from "@react-google-maps/api";

const BASE_URL = "http://localhost:5000/api";

const mapContainerStyle = {
  width: "100%",
  height: "400px",
};

const center = {
  lat: 20.5937,
  lng: 78.9629,
};

// DonationForm Component
const DonationForm = ({ formData, handleInputChange, handleSubmit, isSubmitting, locations }) => (
  <form onSubmit={handleSubmit} className="flex flex-col gap-4 mb-8">
    <input
      type="text"
      name="name"
      placeholder="Your Name"
      value={formData.name}
      onChange={handleInputChange}
      className="border-gray-light border p-2 rounded-md"
      required
    />
    <input
      type="email"
      name="email"
      placeholder="Your Email"
      value={formData.email}
      onChange={handleInputChange}
      className="border-gray-light border p-2 rounded-md"
      required
    />
    <select
      name="supplyType"
      value={formData.supplyType}
      onChange={handleInputChange}
      className="border-gray-light border p-2 rounded-md"
      required
    >
      <option value="" disabled>
        Select Supply Type
      </option>
      <option value="Food">Food</option>
      <option value="Clothes">Clothes</option>
      <option value="Medicines">Medicines</option>
      <option value="Equipment">Equipment</option>
      <option value="Other">Other</option>
    </select>
    <select
      name="dropOffLocation"
      value={formData.dropOffLocation}
      onChange={handleInputChange}
      className="border-gray-light border p-2 rounded-md"
      required
    >
      <option value="" disabled>
        Select Drop-Off Location
      </option>
      {locations.map((location) => (
        
        <option key={location._id} value={location._id}>
          {location.name} - {location.address}
        </option>
      ))}
    </select>
    <textarea
      name="message"
      placeholder="Leave a message (optional)"
      value={formData.message}
      onChange={handleInputChange}
      className="border-gray-light border p-2 rounded-md"
    />
    <div>
      <label className="text-gray">Upload a Photo of Your Supplies (optional):</label>
      <input
        type="file"
        name="photo"
        accept="image/*"
        onChange={handleInputChange}
        className="block mt-2"
      />
    </div>
    <button
      type="submit"
      className={`bg-green text-white px-6 py-2 rounded-md ${
        isSubmitting ? "opacity-50 cursor-not-allowed" : "hover:bg-teal-dark"
      }`}
      disabled={isSubmitting}
    >
      {isSubmitting ? "Submitting..." : "Donate Supplies"}
    </button>
  </form>
);

// MapWithLocations Component
const MapWithLocations = ({ locations, selectedLocation, setSelectedLocation }) => {
  if (!locations || locations.length === 0) {
    return <p>No locations available.</p>; // Handle empty or undefined locations
  }

  return (
    <GoogleMap mapContainerStyle={mapContainerStyle} center={center} zoom={5}>
      {locations.map((location) => {
        if (
          location.coordinates &&
          typeof location.coordinates.lat === "number" &&
          typeof location.coordinates.lng === "number"
        ) {
          return (
            <Marker
              key={location._id}
              position={{
                lat: location.coordinates.lat,
                lng: location.coordinates.lng,
              }}
              onClick={() => setSelectedLocation(location)}
            />
          );
        } else {
          console.warn("Invalid marker data:", location);
          return null;
        }
      })}

      {selectedLocation && (
        <InfoWindow
          position={{
            lat: selectedLocation.coordinates.lat,
            lng: selectedLocation.coordinates.lng,
          }}
          onCloseClick={() => setSelectedLocation(null)}
        >
          <div>
            <h3 className="text-navy font-bold">{selectedLocation.name}</h3>
            <p className="text-gray">{selectedLocation.address}</p>
            {selectedLocation.hours && <p className="text-gray">Hours: {selectedLocation.hours}</p>}
            {selectedLocation.contact && <p className="text-gray">Contact: {selectedLocation.contact}</p>}
          </div>
        </InfoWindow>
      )}
    </GoogleMap>
  );
};


// Main Component
const DonateSupplies = ({ onSuppliesUpdate }) => {
  const [formData, setFormData] = useState({
    name: "",
    email: "",
    supplyType: "",
    dropOffLocation: "",
    message: "",
    photo: null,
  });
  const [locations, setLocations] = useState([]);
  const [selectedLocation, setSelectedLocation] = useState(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [successMessage, setSuccessMessage] = useState("");
  const [error, setError] = useState("");

  const googleMapsApiKey = import.meta.env.VITE_GOOGLE_MAPS_API_KEY;

  const { isLoaded } = useJsApiLoader({
    googleMapsApiKey: googleMapsApiKey,
  });

  useEffect(() => {
    const fetchLocations = async () => {
      try {
        const response = await axios.get(`${BASE_URL}/dropOffLocations`); 
        setLocations(response.data);
      } catch (err) {
        console.error("Error fetching drop-off locations:", err);
      }
    };
    fetchLocations();
  }, []);

  const handleInputChange = (e) => {
    const { name, value, type, files } = e.target;
    setFormData({ ...formData, [name]: type === "file" ? files[0] : value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsSubmitting(true);
    setError("");
    setSuccessMessage("");

    try {
      let photoId = null;
      if (formData.photo) {
        const photoFormData = new FormData();
        photoFormData.append("file", formData.photo);
        const uploadResponse = await axios.post(`${BASE_URL}/uploads`, photoFormData, {
          headers: { "Content-Type": "multipart/form-data" },
        });
        photoId = uploadResponse.data.id;
      }
      const donationData = { ...formData, photoId };
      const response = await axios.post(`${BASE_URL}/supplyDonations`, donationData); // Updated endpoint
      onSuppliesUpdate(response.data);
      setSuccessMessage("Thank you for your donation!");
      setFormData({ name: "", email: "", supplyType: "", dropOffLocation: "", message: "", photo: null });
    } catch (err) {
      setError("There was an error submitting your donation.");
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="donate-supplies-section bg-teal-light p-8 rounded-lg shadow-lg">
      <p className="text-gray mb-6">
        Your donation of essential supplies helps us support those in need.
      </p>
      {successMessage && <p className="text-green mb-4">{successMessage}</p>}
      {error && <p className="text-red mb-4">{error}</p>}
      <DonationForm
        formData={formData}
        handleInputChange={handleInputChange}
        handleSubmit={handleSubmit}
        isSubmitting={isSubmitting}
        locations={locations}
      />
      <div className="map-container">{isLoaded ? <MapWithLocations locations={locations} selectedLocation={selectedLocation} setSelectedLocation={setSelectedLocation} /> : <p>Loading map...</p>}</div>
    </div>
  );
};

export default DonateSupplies;
