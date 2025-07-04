import React from 'react';

const ScrollingAlert = ({ alerts }) => {
  // Ensure alerts is always an array
  const safeAlerts = Array.isArray(alerts) ? alerts : [];
  return (
    <div className="relative w-full">
      {/* bg-blue-100 */}
      <div className="overflow-hidden bg-teal-500 text-gray-800 shadow-md">
        {/* Scrolling container */}
        <div className="scroll-container flex animate-marquee">
          {safeAlerts.length > 0 ? (
            safeAlerts.map((alert, index) => {
              // Defensive checks for alert properties
              const formattedDate = alert && alert.dateTime ? new Date(alert.dateTime).toISOString().split('T')[0] : '';
              const key = alert && alert.id ? alert.id : index;
              const address = alert && alert.location && alert.location.address ? alert.location.address.split(',')[0] : 'Unknown Location';
              const type = alert && alert.type ? alert.type : 'Unknown Type';
              return (
                <div
                  key={key}
                  className="flex-shrink-0 px-6 py-2 border-l border-gray-300 first:border-none"
                >
                  <span className="font-semibold text-sm text-white">
                    {address} | {type} | {formattedDate}
                  </span>
                </div>
              );
            })
          ) : (
            <div className="flex-shrink-0 px-6 py-2">
              <span className="font-semibold text-sm">No active alerts</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ScrollingAlert;
