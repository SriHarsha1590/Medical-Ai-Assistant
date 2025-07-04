import React from 'react';
import { disasterDosDonts } from './disasterDosDonts';

const Modal = ({ isOpen, onClose, disaster }) => {
  if (!isOpen) return null;

  const dosAndDonts = disasterDosDonts[disaster.type.toLowerCase()];

  // Helper function to format the date
  const formatDate = (date) => {
    return new Date(date).toLocaleString();
  };

  return (
    <div className="fixed inset-0 flex items-center justify-center bg-gray bg-opacity-75 z-50">
      <div className="bg-white rounded-lg shadow-lg p-6 w-[90%] max-w-2xl text-gray-800">
        <h2 className="text-xl font-bold mb-4 text-teal-500">Disaster Details</h2>

        {/* Basic Details */}
        <p className="mb-2"><strong>Type:</strong> {disaster.type}</p>
        <p className="mb-2"><strong>Severity:</strong> {disaster.severity}</p>
        <p className="mb-2"><strong>Status:</strong> {disaster.status}</p>

        {/* Location and Date */}
        <p className="mb-2"><strong>Location:</strong> {disaster.location?.address || 'N/A'}</p>
        <p className="mb-2"><strong>Date:</strong> {disaster.dateTime ? formatDate(disaster.dateTime) : 'N/A'}</p>

        {/* Status Updates */}
        {disaster.statusUpdates && disaster.statusUpdates.length > 0 && (
          <div className="mb-4">
            <h4 className="font-semibold mt-4 mb-2 text-teal-500">Recent Updates</h4>
            <ul>
              {disaster.statusUpdates.map((update, index) => (
                <li key={index} className="mb-2 text-gray-700">
                  <strong>{formatDate(update.dateTime)}:</strong> {update.message}
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Do's and Don'ts */}
        {dosAndDonts && (
          <>
            <h4 className="font-semibold mt-4 mb-2 text-teal-500">Do's and Don'ts</h4>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <h5 className="font-semibold text-teal-500">Do's</h5>
                <ul className="list-disc ml-5 text-gray-700">
                  {dosAndDonts.do.map((item, index) => (
                    <li key={index}>{item}</li>
                  ))}
                </ul>
              </div>
              <div>
                <h5 className="font-semibold text-red">Don'ts</h5>
                <ul className="list-disc ml-5 text-gray-700">
                  {dosAndDonts.dont.map((item, index) => (
                    <li key={index}>{item}</li>
                  ))}
                </ul>
              </div>
            </div>
          </>
        )}

        {/* Casualties or Impact Summary */}
        {disaster.casualties && (
          <p className="mb-2"><strong>Casualties:</strong> {disaster.casualties} (if confirmed)</p>
        )}

        {/* Close Button */}
        <button
          onClick={onClose}
          className="mt-6 px-4 py-2 bg-teal text-white rounded-md hover:bg-teal-700 transition"
        >
          Close
        </button>
      </div>
    </div>
  );
};

export default Modal;
