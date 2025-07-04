import React, { useEffect, useState } from 'react';
import Button from '../Button/Button';
import { Carousel } from 'react-responsive-carousel'; // Import the carousel component

const DisasterModal = ({ disaster, onClose }) => {
  const [shelterNames, setShelterNames] = useState({});
  const [hospitalNames, setHospitalNames] = useState({});

  if (!disaster) return null;

  // Close modal on outside click
  const handleOutsideClick = (e) => {
    if (e.target.id === 'modalOverlay') {
      onClose();
    }
  };

  // Close modal on 'Escape' key press
  useEffect(() => {
    const handleEscape = (e) => {
      if (e.key === 'Escape') onClose();
    };
    document.addEventListener('keydown', handleEscape);
    return () => {
      document.removeEventListener('keydown', handleEscape);
    };
  }, [onClose]);

  // Prevent click propagation inside the modal
  const handleContentClick = (e) => {
    e.stopPropagation(); // This prevents the click from propagating
  };

  // Fetch shelter data when component mounts
  useEffect(() => {
    if (disaster.shelters && disaster.shelters.length > 0) {
      const shelterIds = disaster.shelters;
      shelterIds.forEach((shelterId) => {
        // Fetch shelter data by ID (assuming API endpoint exists to get shelter info)
        fetch(`/api/shelters/${shelterId}`)
          .then(response => response.json())
          .then(data => {
            setShelterNames(prevState => ({
              ...prevState,
              [shelterId]: data.name,
            }));
          })
          .catch(error => console.error('Error fetching shelter data:', error));
      });
    }
  }, [disaster.shelters]);

  useEffect(() => {
    if (disaster.hospitals && disaster.hospitals.length > 0) {
      const hospitalIds = disaster.hospitals;
      hospitalIds.forEach((shelterId) => {
        // Fetch shelter data by ID (assuming API endpoint exists to get shelter info)
        fetch(`/api/shelters/${shelterId}`)
          .then(response => response.json())
          .then(data => {
            setHospitalNames(prevState => ({
              ...prevState,
              [shelterId]: data.name,
            }));
          })
          .catch(error => console.error('Error fetching shelter data:', error));
      });
    }
  }, [disaster.hospitals]);

  return (
    <div
      id="modalOverlay"
      onClick={handleOutsideClick}
      className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50 z-50"
    >
      <div
        className="bg-white w-5/6 max-w-4xl max-h-[80vh] rounded-lg p-6 shadow-lg relative overflow-y-auto"
        onClick={handleContentClick} // Prevent click from propagating
      >
        {/* Modal Header */}
        <h2 className="text-teal text-2xl font-semibold mb-2 flex justify-between items-center">
          {disaster.type}
          <Button onClick={onClose} variant="danger" size="small">
            &times;
          </Button>
        </h2>

        {/* Modal Content */}
        <p className="text-gray mb-1">Location: {disaster.location?.address || 'N/A'}</p>
        <p className="mb-2">Date: {disaster.dateTime ? new Date(disaster.dateTime.$date).toLocaleString() : 'N/A'}</p>
        <p className="mb-2">Severity: {disaster.severity || 'N/A'}</p>
        <p className="mb-4">Description: {disaster.description || 'No description available'}</p>
        <p className="mb-4">Affected Areas: {disaster.affectedAreas ? disaster.affectedAreas.join(', ') : 'N/A'}</p>

        {/* Media Images and Videos */}
        {(disaster.media?.images?.length > 0 || disaster.media?.videos?.length > 0) && (
          <div>
            <h4 className="text-lg font-semibold mt-4 mb-2">Media</h4>
            <Carousel showThumbs={false} dynamicHeight={true}>
              {disaster.media.images && disaster.media.images.map((image, index) => (
                <div key={index}>
                  <img src={image} alt={`${disaster.type} media ${index + 1}`} className="rounded max-w-full" />
                </div>
              ))}
              {disaster.media.videos && disaster.media.videos.map((video, index) => (
                <div key={index}>
                  <video controls className="rounded max-w-full">
                    <source src={video} type="video/mp4" />
                    Your browser does not support the video tag.
                  </video>
                </div>
              ))}
            </Carousel>
          </div>
        )}

        {/* Shelters */}
        {disaster.shelters && disaster.shelters.length > 0 && (
          <div>
            <h4 className="text-lg font-semibold mt-4 mb-2">Shelters</h4>
            {disaster.shelters.map((shelterId, index) => (
              <p key={index} className="mb-2 text-teal-600">
                <a href={`/shelter/${shelterId}`} className="hover:underline">
                  Shelter: {shelterNames[shelterId] || 'Loading...'}
                </a>
              </p>
            ))}
          </div>
        )}

        {disaster.hospitals && disaster.hospitals.length > 0 && (
          <div>
            <h4 className="text-lg font-semibold mt-4 mb-2">Hospitals</h4>
            {disaster.hospitals.map((shelterId, index) => (
              <p key={index} className="mb-2 text-teal-600">
                <a href={`/shelter/${shelterId}`} className="hover:underline">
                  hospital: {hospitalNames[shelterId] || 'Loading...'}
                </a>
              </p>
            ))}
          </div>
        )}

        {/* Status Updates */}
        {disaster.statusUpdates && disaster.statusUpdates.length > 0 && (
          <div>
            <h4 className="text-lg font-semibold mt-4 mb-2">Status Updates</h4>
            {disaster.statusUpdates.map((update, index) => {
              // Safeguard for date format
              const formattedDate = update.dateTime ? new Date(update.dateTime).toLocaleString() : 'Date not available';
              return (
                <div key={index} className="mb-2">
                  <p className="text-gray-700">
                    <strong>{formattedDate}:</strong> {update.message}
                  </p>
                </div>
              );
            })}
          </div>
        )}


        {/* Emergency Contacts */}
        {disaster.emergencyContacts && disaster.emergencyContacts.length > 0 && (
          <div>
            <h4 className="text-lg font-semibold mt-4 mb-2">Emergency Contacts</h4>
            {disaster.emergencyContacts.map((contact, index) => (
              <div key={index} className="mb-2">
                <p>Name: {contact.name}</p>
                <p>Phone: {contact.phone}</p>
                <p>Role: {contact.role}</p>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default DisasterModal;
