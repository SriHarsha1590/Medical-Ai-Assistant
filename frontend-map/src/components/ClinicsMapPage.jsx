import React, { useEffect, useState } from 'react';
import MapView from './MapView';

const ClinicsMapPage = () => {
  const [clinics, setClinics] = useState([]);
  const [userLocation, setUserLocation] = useState(null);

  useEffect(() => {
    // Fetch clinics from Flask backend
    fetch('http://localhost:5000/api/clinics')
      .then(res => res.json())
      .then(data => setClinics(data))
      .catch(() => setClinics([]));

    // Get user location
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          setUserLocation({
            lat: position.coords.latitude,
            lng: position.coords.longitude,
          });
        },
        () => setUserLocation(null)
      );
    }
  }, []);

  return (
    <div className="container mx-auto py-8">
      <h1 className="text-2xl font-bold mb-4 text-center">Nearby Clinics & Hospitals</h1>
      <div className="flex justify-center">
        <div className="w-full max-w-4xl">
          <MapView clinics={clinics} userLocation={userLocation} />
        </div>
      </div>
    </div>
  );
};

export default ClinicsMapPage;
