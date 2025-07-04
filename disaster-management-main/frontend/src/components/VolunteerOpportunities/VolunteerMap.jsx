import React, { useState } from 'react';
import { GoogleMap, Marker, InfoWindow, useJsApiLoader } from '@react-google-maps/api';

const googleMapsApiKey = import.meta.env.VITE_GOOGLE_MAPS_API_KEY;

const VolunteerMap = ({ opportunities }) => {
  // State to handle which marker's info window is open
  const [selectedMarker, setSelectedMarker] = useState(null);

  // Load Google Maps API using the `useJsApiLoader` hook
  const { isLoaded } = useJsApiLoader({
    googleMapsApiKey: googleMapsApiKey,
  });

  // Set default center and zoom level for the map
  const defaultCenter = { lat: 20.5937, lng: 78.9629 }; // Center of India
  const defaultZoom = 5; // A zoom level that shows India well

  if (!isLoaded) {
    return <div>Loading...</div>; // Show loading message while the map is loading
  }

  return (
    <div className="w-full h-64 bg-gray">
      <GoogleMap
        mapContainerStyle={{ width: '100%', height: '400px' }}
        center={defaultCenter}
        zoom={defaultZoom}
      >
        {opportunities.map((opportunity) => {
  const { coordinates } = opportunity.location || {};
  if (coordinates && typeof coordinates.lat === "number" && typeof coordinates.lng === "number") {
    return (
      <Marker
        key={opportunity._id}
        position={{ lat: coordinates.lat, lng: coordinates.lng }}
        onClick={() => setSelectedMarker(opportunity)}
      />
    );
  } else {
    console.warn("Invalid marker data:", opportunity);
    return null;
  }
})}


        {/* InfoWindow for the selected marker */}
        {selectedMarker && (
          <InfoWindow
            position={{
              lat: selectedMarker.location.coordinates.lat,
              lng: selectedMarker.location.coordinates.lng,
            }}
            onCloseClick={() => setSelectedMarker(null)} // Close the InfoWindow
          >
            <div>
              <h3>{selectedMarker.title}</h3>
              <p>{selectedMarker.location.address}</p>
            </div>
          </InfoWindow>
        )}
      </GoogleMap>
    </div>
  );
};

export default VolunteerMap;


