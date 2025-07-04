import React, { useState } from "react";
import Button from "../Button/Button";
import Filter from "./Filter";  // Assuming the Filter component is imported here
import { toast } from "react-toastify"; // Import toast
import "react-toastify/dist/ReactToastify.css"; // Import toast styles

const Header = ({
  disasters,
  setIsMapView,
  isMapView,
  refreshFeed,
  filters,
  setFilters,
  userLocation,
  setUserLocation,
  onApplyFilters
}) => {
  const [isFilterVisible, setIsFilterVisible] = useState(false);
  const [loadingLocation, setLoadingLocation] = useState(false);

  const toggleView = () => {
    setIsMapView(!isMapView);
  };

  const getUserLocation = () => {
    if (navigator.geolocation) {
      setLoadingLocation(true);
      navigator.geolocation.getCurrentPosition(
        (position) => {
          const newLocation = {
            latitude: position.coords.latitude,
            longitude: position.coords.longitude,
          };
          setUserLocation(newLocation);
          toast.success(`Your location is: Latitude: ${newLocation.latitude}, Longitude: ${newLocation.longitude}`);
          setLoadingLocation(false);
        },
        (error) => {
          toast.error("Error getting location: " + error.message);
          setLoadingLocation(false);
        }
      );
    } else {
      toast.error("Geolocation is not supported by this browser.");
      setLoadingLocation(false);
    }
  };

  // Haversine formula to calculate distance between two lat/lon points
  const getDistance = (lat1, lon1, lat2, lon2) => {
    const R = 6371; // Earth radius in kilometers
    const dLat = ((lat2 - lat1) * Math.PI) / 180;
    const dLon = ((lon2 - lon1) * Math.PI) / 180;
    const a =
      Math.sin(dLat / 2) * Math.sin(dLat / 2) +
      Math.cos((lat1 * Math.PI) / 180) *
        Math.cos((lat2 * Math.PI) / 180) *
        Math.sin(dLon / 2) * Math.sin(dLon / 2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
    const distance = R * c; // Distance in kilometers
    return distance;
  };

  // Check if user is within red zone for any disaster
  const checkRedZone = () => {
    if (!userLocation) {
      toast.error("Please get your location first.");
      return;
    }

    // Iterate over disasters to check if user is within the red zone
    let inRedZone = false;
    disasters.forEach((disaster) => {
      if (disaster.redZoneRange && disaster.location) {
        const distance = getDistance(
          userLocation.latitude,
          userLocation.longitude,
          disaster.location.latitude,
          disaster.location.longitude
        );

        // Check if the user is within the red zone range
        if (distance <= disaster.redZoneRange) {
          inRedZone = true;
          toast.info(`You are within the red zone of the ${disaster.type} disaster!`);
        }
      }
    });

    if (!inRedZone) {
      toast.success("You are not within any red zone.");
    }
  };

  return (
    <header className="flex flex-col md:flex-row justify-between items-center p-4 bg-navy text-white">
      {/* Left Buttons: View Toggle, Location, Red Zone */}
      <div className="flex flex-wrap gap-4 justify-center mb-4 md:mb-0">
        <Button onClick={toggleView} variant="success" size="medium">
          {isMapView ? "Switch to List View" : "Switch to Map View"}
        </Button>
        <Button onClick={getUserLocation} variant="success" size="medium" disabled={loadingLocation}>
          {loadingLocation ? "Getting Location..." : "Get My Location"}
        </Button>
        <Button onClick={checkRedZone} variant="danger" size="medium">
          Check Red Zone
        </Button>
      </div>

      {/* Right Buttons: Filter Toggle, Refresh Feed */}
      <div className="flex flex-wrap gap-4 justify-center mb-4 md:mb-0">
        <Button
          onClick={() => setIsFilterVisible(!isFilterVisible)} // Toggle filter visibility
          variant="secondary"
          size="medium"
        >
          {isFilterVisible ? "Close Filters" : "Filter Disasters"}
        </Button>
        <Button onClick={refreshFeed} variant="secondary" size="medium">
          Refresh Feed
        </Button>
      </div>

      {/* Conditional Rendering of Filter Modal */}
      {isFilterVisible && (
        <div
          className="fixed inset-0 flex items-center justify-center z-50 bg-black bg-opacity-50"
          role="dialog"
          aria-labelledby="filter-modal-title"
          aria-modal="true"
        >
          <div className="relative w-11/12 sm:w-96 p-6 bg-white shadow-xl rounded-lg border-2 border-gray-200">
            <div className="absolute top-3 right-3">
              <Button
                className="text-xl text-gray-500 hover:text-gray-700"
                onClick={() => setIsFilterVisible(false)} // Close modal
                aria-label="Close filter modal"
              >
                &times;
              </Button>
            </div>

            <h3 id="filter-modal-title" className="text-2xl font-semibold text-navy mb-4 text-center">
              Disaster Filters
            </h3>

            <Filter
              filters={filters}
              onFilterChange={setFilters}
              onApplyFilters={onApplyFilters}
              setIsFilterVisible={setIsFilterVisible} // Pass down the setter to the Filter
            />
          </div>
        </div>
      )}
    </header>
  );
};

export default Header;
