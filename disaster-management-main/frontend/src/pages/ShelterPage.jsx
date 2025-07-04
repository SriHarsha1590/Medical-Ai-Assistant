import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { getAllHospitals } from "../services/hospitalService";
import { getAllShelters } from "../services/shelterService";
import Filter from "../components/ShelterPage/Filter";
import Button from "../components/Button/Button";
import { MdSearch, MdFilterList } from "react-icons/md";
import "../styles/card.css";

const ShelterPage = () => {
  const [items, setItems] = useState([]);
  const [allItems, setAllItems] = useState([]);
  const [searchLocation, setSearchLocation] = useState("");
  const [disasterSearch, setDisasterSearch] = useState("");
  const [selectedType, setSelectedType] = useState("shelter");
  const [isFilterVisible, setIsFilterVisible] = useState(false);
  const [filters, setFilters] = useState({
    availability: "all",
    type: "all",
    nearMe: false,
    facilities: [], // Added to store selected facilities
    services: [], // Added to store selected services
    emergencyBeds: "all", // Ensures compatibility with hospital filters
    availabilityStatus: "all", // Ensures compatibility with hospital filters
  });
  
  const [userLocation, setUserLocation] = useState(null);

  const navigate = useNavigate();

  useEffect(() => {
    if (filters.nearMe) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          setUserLocation({
            lat: position.coords.latitude,
            lon: position.coords.longitude,
          });
        },
        (error) => console.error("Geolocation Error:", error),
        { enableHighAccuracy: true }
      );
    }
  }, [filters.nearMe]);

  useEffect(() => {
    const fetchItems = async () => {
      const sheltersData = await getAllShelters();
      const hospitalsData = await getAllHospitals();
      const data = selectedType === "shelter" ? sheltersData : hospitalsData;
      setItems(data);
      setAllItems(data);
    };
    fetchItems();
  }, [selectedType]);

  useEffect(() => {
    let filteredItems = allItems;
  
    // Search by location
    if (searchLocation) {
      filteredItems = filteredItems.filter(
        (item) =>
          item.name.toLowerCase().includes(searchLocation.toLowerCase()) ||
          (item.location?.address &&
            item.location.address.toLowerCase().includes(searchLocation.toLowerCase()))
      );
    }
  
    // Search by disaster
    if (disasterSearch) {
      filteredItems = filteredItems.filter((item) =>
        item.affiliatedDisasters.some(
          (disaster) =>
            disaster.type.toLowerCase().includes(disasterSearch.toLowerCase()) ||
            disaster.location?.address?.toLowerCase().includes(disasterSearch.toLowerCase())
        )
      );
    }
  
    // Filter by availability
    if (filters.availability === "available") {
      filteredItems = filteredItems.filter((item) => item.available < item.capacity);
    }
  
    // Shelter-specific filters
    if (selectedType === "shelter") {
      // Filter by type
      if (filters.type !== "all") {
        filteredItems = filteredItems.filter((item) => item.type === filters.type);
      }
  
      // Filter by facilities (must include all selected facilities)
      if (filters.facilities.length > 0) {
        filteredItems = filteredItems.filter((item) =>
          filters.facilities.every((facility) => item.facilities.includes(facility))
        );
      }
  
      // Filter by services (must include all selected services)
      if (filters.services.length > 0) {
        filteredItems = filteredItems.filter((item) =>
          filters.services.every((service) => item.services.includes(service))
        );
      }
    }
  
    // Hospital-specific filters
    if (selectedType === "hospital") {
      // Filter by availability status
      if (filters.availabilityStatus !== "all") {
        filteredItems = filteredItems.filter(
          (item) => item.availabilityStatus === filters.availabilityStatus
        );
      }
  
      // Filter by emergency beds
      if (filters.emergencyBeds !== "all") {
        const minBeds = parseInt(filters.emergencyBeds, 10);
        filteredItems = filteredItems.filter((item) => item.emergencyBeds >= minBeds);
      }
    }
  
    // Filter by "Near Me" (geolocation)
    if (filters.nearMe && userLocation) {
      filteredItems = filteredItems.filter((item) => {
        const itemLocation = item.location;
        if (itemLocation?.coordinates) {
          const [itemLat, itemLon] = itemLocation.coordinates;
          const distance = calculateDistance(
            userLocation.lat,
            userLocation.lon,
            itemLat,
            itemLon
          );
          return distance <= 50; // Example: within 50 km
        }
        return false;
      });
    }
  
    setItems(filteredItems);
  }, [searchLocation, disasterSearch, allItems, selectedType, filters, userLocation]);
  

  const calculateDistance = (lat1, lon1, lat2, lon2) => {
    const R = 6371;
    const dLat = ((lat2 - lat1) * Math.PI) / 180;
    const dLon = ((lon2 - lon1) * Math.PI) / 180;
    const a =
      Math.sin(dLat / 2) * Math.sin(dLat / 2) +
      Math.cos((lat1 * Math.PI) / 180) *
        Math.cos((lat2 * Math.PI) / 180) *
        Math.sin(dLon / 2) * Math.sin(dLon / 2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
    return R * c;
  };

  const handleCardClick = (item) => {
    navigate(`/${selectedType}/${item._id}`);
  };

  return (
    <div className="container mx-auto px-4 py-6">
      <div className="flex justify-between items-center mb-6">
        <div className="flex items-center space-x-4">
          <button
            onClick={() => setSelectedType("shelter")}
            className={`${selectedType === "shelter" ? "text-blue-500 font-bold" : "text-gray-700"}`}
          >
            Shelters
          </button>
          <button
            onClick={() => setSelectedType("hospital")}
            className={`${selectedType === "hospital" ? "text-blue-500 font-bold" : "text-gray-700"}`}
          >
            Hospitals
          </button>
        </div>

        <div className="flex items-center space-x-4">
          <div className="relative">
            <input
              type="text"
              placeholder="Search by location"
              value={searchLocation}
              onChange={(e) => setSearchLocation(e.target.value)}
              className="border rounded-md pl-10 py-2 w-full"
            />
            <MdSearch className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
          </div>
          <div className="relative">
            <input
              type="text"
              placeholder="Search by disaster related"
              value={disasterSearch}
              onChange={(e) => setDisasterSearch(e.target.value)}
              className="border rounded-md pl-10 py-2 w-full"
            />
            <MdSearch className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
          </div>
          <Button
            variant="secondary"
            className="bg-blue-500 text-white px-4 py-2 rounded-md"
            onClick={() => setIsFilterVisible(true)}
          >
            <MdFilterList className="inline-block mr-2" />
            Filter
          </Button>
        </div>
      </div>

      {isFilterVisible && (
        <Filter
          selectedType={selectedType}
          filters={filters}
          setFilters={setFilters}
          closeModal={() => setIsFilterVisible(false)}
        />
      )}

      <div className="card-grid">
        {items.map((item) => (
          <div
            key={item.id}
            className="card"
            onClick={() => handleCardClick(item)}
          >
            <img
              src={
                item.photo
                  ? `http://localhost:5000/api/uploads/${item.photo}`
                  : "https://via.placeholder.com/150"
              }
              alt={item.name}
              className="card-image"
            />
            <div className="card-content">
              <h3>{item.name}</h3>
              <p>{item.location?.address}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default ShelterPage;
