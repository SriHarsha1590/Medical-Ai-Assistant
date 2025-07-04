import React from 'react';

const DisasterCard = ({ disaster, onCardClick }) => {
  // Dynamic severity color based on disaster severity
  const severityColor = disaster.severity === 'Severe' ? 'red' :
                        disaster.severity === 'Moderate' ? 'yellow' :
                        disaster.severity === 'High' ? 'black' : 'orange-500';

  return (
    <div
      className="flex flex-col bg-white border border-gray-300 rounded-lg p-4 mb-4 h-72 cursor-pointer hover:shadow-lg transition-shadow duration-200 overflow-hidden"
      onClick={() => onCardClick(disaster)} // Trigger the modal or expanded view
    >
      {/* Disaster Type */}
      <h3 className="text-teal-600 text-xl font-semibold mb-2">{disaster.type}</h3>
      
      {/* Location */}
      <p className="text-gray-600 text-sm mb-2">{disaster.location.address}</p>
      
      {/* Date of Occurrence */}
      <p className="text-gray-500 text-sm mb-2">{new Date(disaster.dateTime).toLocaleDateString()}</p>
      
      {/* Severity */}
      <p className={`font-bold text-${severityColor} mb-2`}>
        Severity: {disaster.severity}
      </p>

      {/* Description */}
      <p className="text-gray-600 mb-4 overflow-hidden text-ellipsis">
        {disaster.description || "No description available."}
      </p>
    </div>
  );
};

export default DisasterCard;
