import React, { useState, useEffect, useMemo } from 'react';
import Header from '../components/DisasterFeed/Header';
import WeatherWidget from '../components/DisasterFeed/WeatherWidget';
import DisasterCard from '../components/DisasterFeed/DisasterCard';
import MapView from '../components/DisasterFeed/Mapview';
import DisasterModal from '../components/DisasterFeed/DisasterModal'; // Import the modal component

const DisasterFeed = ({ disasters, refreshDisasters }) => {
  const [isMapView, setIsMapView] = useState(true);
  const [weather, setWeather] = useState(null);
  const [loadingWeather, setLoadingWeather] = useState(true);
  const [userLocation, setUserLocation] = useState(null);
  const [filters, setFilters] = useState({
    type: "All",
    severity: "All",
    status: "All",
  });
  const [loadingDisasters, setLoadingDisasters] = useState(false);
  const [selectedDisaster, setSelectedDisaster] = useState(null); // Track selected disaster for modal

  // Fetch weather data based on user latitude and longitude
  const fetchWeather = async (latitude, longitude) => {
    setLoadingWeather(true);
    try {
      const apiKey = "90b950dba316f5e49546655bb9ae5df7";
      const res = await fetch(
        `https://api.openweathermap.org/data/2.5/weather?lat=${latitude}&lon=${longitude}&appid=${apiKey}&units=metric`
      );
      if (!res.ok) throw new Error('Network response was not ok');
      const data = await res.json();
      setWeather(data);
    } catch (error) {
      console.error("Error fetching weather data:", error.message);
    } finally {
      setLoadingWeather(false);
    }
  };

  // Get the user's location and set it
  useEffect(() => {
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          const { latitude, longitude } = position.coords;
          setUserLocation({ latitude, longitude });
        },
        (error) => {
          console.error("Error getting location:", error.message);
        }
      );
    }
  }, []);

  // Fetch weather data when userLocation is updated
  useEffect(() => {
    if (userLocation) {
      fetchWeather(userLocation.latitude, userLocation.longitude);
    }
  }, [userLocation]);

  // Filter disasters based on selected filters
  const displayedDisasters = useMemo(() => {
    return disasters.filter((disaster) => {
      const typeMatch = filters.type === "All" || disaster.type === filters.type;
      const severityMatch = filters.severity === "All" || disaster.severity === filters.severity;
      const statusMatch = filters.status === "All" || disaster.status === filters.status;
      return typeMatch && severityMatch && statusMatch;
    });
  }, [disasters, filters]);

  // Open modal with selected disaster
  const handleCardClick = (disaster) => {
    setSelectedDisaster(disaster);
  };

  // Close the modal
  const handleCloseModal = () => {
    setSelectedDisaster(null);
  };

  return (
    <div className="disaster-feed-page flex flex-col min-h-screen m-4 bg-gray-light">
      <Header
        disasters={disasters}
        setIsMapView={setIsMapView}
        isMapView={isMapView}
        refreshFeed={refreshDisasters}
        filters={filters}
        setFilters={setFilters}
        userLocation={userLocation}
        setUserLocation={setUserLocation}
      />

      <div className="flex flex-row mt-4">
        {/* MapView */}
        <div
          className={`map-view flex-1 overflow-y-auto p-4 ${isMapView ? 'block' : 'hidden'}`}
          style={{ height: '80vh' }}
        >
          <MapView disasters={displayedDisasters} userLocation={userLocation} />
        </div>

        {/* List view */}
        <div
          className={`list-view flex-1 grid gap-4 p-4 ${!isMapView ? 'block' : 'hidden'}`}
          style={{
            overflowY: 'hidden',
            gridTemplateColumns: 'repeat(auto-fill, minmax(250px, 1fr))',
            gridAutoRows: 'auto',
          }}
        >
          {loadingDisasters ? (
            <p className="text-teal">Loading disasters...</p>
          ) : displayedDisasters.length === 0 ? (
            <p className="text-teal">No disasters data available.</p>
          ) : (
            displayedDisasters.map((disaster, index) => (
              <div
                key={index}
                className="transition-all duration-300 ease-in-out transform hover:scale-105 hover:rounded-lg"
              >
                <DisasterCard disaster={disaster} onCardClick={handleCardClick} />
              </div>
            ))
          )}
        </div>

        {/* Weather Widget */}
        <div
          className="weather-widget-container flex-shrink-0 p-4 w-64 text-white rounded-lg "
          style={{
            position: 'sticky',
            top: '20px',
            height: 'calc(80vh - 10px)',
          }}
        >
          {loadingWeather ? (
            <p className="text-teal">Loading weather...</p>
          ) : weather ? (
            <WeatherWidget weather={weather} />
          ) : (
            <p className="text-teal">Unable to fetch weather data.</p>
          )}
        </div>
      </div>

      {/* Modal */}
      {selectedDisaster && (
        <DisasterModal disaster={selectedDisaster} onClose={handleCloseModal} />
      )}
    </div>
  );
};

export default DisasterFeed;
