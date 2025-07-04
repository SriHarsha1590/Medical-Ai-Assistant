import React, { useEffect, useRef, useState } from 'react';
import { LoadScriptNext } from '@react-google-maps/api';
import Modal from './Modal';

const googleMapsApiKey = import.meta.env.VITE_GOOGLE_MAPS_API_KEY;

const MapView = ({ disasters, userLocation }) => {
  const mapRef = useRef(null);
  const mapInstance = useRef(null);

  const [modalInfo, setModalInfo] = useState(null);

  const initMap = () => {
    if (!mapRef.current) return;

    mapInstance.current = new window.google.maps.Map(mapRef.current, {
      center: userLocation
        ? { lat: userLocation.latitude, lng: userLocation.longitude }
        : { lat: 20.5937, lng: 78.9629 },
      zoom: 5,
      mapId: '1ZAt_cPnU9Ldybq7Ooff6Cnkud6j9gm0',
    });

    renderMarkers();
  };

  const renderMarkers = () => {
    if (!mapInstance.current) return;

    mapInstance.current.markers = mapInstance.current.markers || [];
    mapInstance.current.markers.forEach(marker => marker.setMap(null));
    mapInstance.current.markers = [];

    disasters.forEach(disaster => {
      const { coordinates } = disaster.location;
      const [latitude, longitude] = coordinates;

      if (typeof latitude === 'number' && !isNaN(latitude) && typeof longitude === 'number' && !isNaN(longitude)) {
        const marker = new window.google.maps.Marker({
          map: mapInstance.current,
          position: { lat: latitude, lng: longitude },
          title: disaster.location.address,
        });

        const infoWindow = new window.google.maps.InfoWindow({
          content: `<div>Type: ${disaster.type}<br>Severity: ${disaster.severity}<br>Status: ${disaster.status}</div>`,
        });

        marker.addListener('mouseover', () => {
          infoWindow.open(mapInstance.current, marker);
        });

        marker.addListener('mouseout', () => {
          infoWindow.close();
        });

        marker.addListener('click', () => {
          setModalInfo(disaster);
        });

        mapInstance.current.markers.push(marker);
      }
    });
  };

  useEffect(() => {
    if (window.google && mapRef.current) {
      if (!mapInstance.current) {
        initMap();
      } else {
        mapInstance.current.setCenter(
          userLocation
            ? { lat: userLocation.latitude, lng: userLocation.longitude }
            : { lat: 20.5937, lng: 78.9629 }
        );
        renderMarkers();
      }
    }
  }, [userLocation, disasters]);

  return (
    <LoadScriptNext googleMapsApiKey={googleMapsApiKey}>
      <div id="map" ref={mapRef} style={{ width: '100%', height: '100%' }} />

      <Modal 
        isOpen={!!modalInfo} 
        onClose={() => setModalInfo(null)} 
        disaster={modalInfo} 
      />
    </LoadScriptNext>
  );
};

export default MapView;
