import React, { useState, useMemo } from 'react';
import { GoogleMap, Marker, InfoWindow, useJsApiLoader } from '@react-google-maps/api';

const googleMapsApiKey = import.meta.env.VITE_GOOGLE_MAPS_API_KEY;

// Custom map style: blue roads, clear outlines, non-bold black place names
const customMapStyles = [
  { elementType: 'geometry', stylers: [{ color: '#f5f5f5' }] },
  { elementType: 'labels.text.fill', stylers: [{ color: '#222', fontWeight: 'normal' }] },
  { elementType: 'labels.text.stroke', stylers: [{ color: '#fff' }] },
  { featureType: 'road', elementType: 'geometry', stylers: [{ color: '#90caf9' }] },
  { featureType: 'road', elementType: 'labels.text.fill', stylers: [{ color: '#222', fontWeight: 'normal' }] },
  { featureType: 'road', elementType: 'labels.text.stroke', stylers: [{ color: '#fff' }] },
  { featureType: 'poi', elementType: 'geometry', stylers: [{ color: '#e0e0e0' }] },
  { featureType: 'poi', elementType: 'labels.text.fill', stylers: [{ color: '#222', fontWeight: 'normal' }] },
  { featureType: 'poi', elementType: 'labels.text.stroke', stylers: [{ color: '#fff' }] },
  { featureType: 'water', elementType: 'geometry', stylers: [{ color: '#b3e5fc' }] },
  { featureType: 'water', elementType: 'labels.text.fill', stylers: [{ color: '#222', fontWeight: 'normal' }] },
  { featureType: 'water', elementType: 'labels.text.stroke', stylers: [{ color: '#fff' }] },
];

const MapView = ({ locations = [], userLocation = null, markerType = 'default' }) => {
  const [selected, setSelected] = useState(null);
  const { isLoaded } = useJsApiLoader({ googleMapsApiKey });

  // Center on first location or user location
  const center = useMemo(() => {
    if (locations.length > 0 && locations[0].lat && locations[0].lng) {
      return { lat: locations[0].lat, lng: locations[0].lng };
    }
    if (userLocation && userLocation.lat && userLocation.lng) {
      return { lat: userLocation.lat, lng: userLocation.lng };
    }
    return { lat: 22.9734, lng: 78.6569 }; // India default
  }, [locations, userLocation]);

  if (!isLoaded) return <div>Loading map...</div>;

  return (
    <GoogleMap
      mapContainerStyle={{ width: '100%', height: '400px', borderRadius: '16px', boxShadow: '0 2px 24px rgba(0,198,255,0.13)' }}
      center={center}
      zoom={locations.length > 0 ? 12 : 5}
      options={{ styles: customMapStyles, mapTypeControl: false, streetViewControl: false, fullscreenControl: true, zoomControl: true }}
    >
      {/* Markers for locations (clinics, hospitals, shelters, etc.) */}
      {locations.map((loc, idx) =>
        loc.lat && loc.lng ? (
          <Marker
            key={loc._id || idx}
            position={{ lat: loc.lat, lng: loc.lng }}
            icon={markerType === 'hospital' ? {
              url: 'https://maps.google.com/mapfiles/ms/icons/hospitals.png',
              scaledSize: { width: 32, height: 32 }
            } : undefined}
            onClick={() => setSelected(loc)}
            label={{ text: String(idx + 1), color: '#222', fontWeight: 'normal', fontSize: '12px' }}
          />
        ) : null
      )}
      {/* User location marker */}
      {userLocation && userLocation.lat && userLocation.lng && (
        <Marker
          position={userLocation}
          icon={{
            url: 'https://maps.google.com/mapfiles/ms/icons/blue-dot.png',
            scaledSize: { width: 32, height: 32 }
          }}
          title="Your Location"
        />
      )}
      {/* InfoWindow for selected marker */}
      {selected && (
        <InfoWindow
          position={{ lat: selected.lat, lng: selected.lng }}
          onCloseClick={() => setSelected(null)}
        >
          <div>
            <strong>{selected.name}</strong>
            <br />
            {selected.address}
          </div>
        </InfoWindow>
      )}
    </GoogleMap>
  );
};

export default MapView;
