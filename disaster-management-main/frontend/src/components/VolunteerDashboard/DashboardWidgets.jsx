// src/components/DashboardWidgets.jsx
import React from "react";

const DashboardWidgets = () => {
  return (
    <div>
      <h2 className="text-2xl font-bold text-navy mb-4">Welcome, Volunteer!</h2>
      <div className="grid grid-cols-3 gap-6">
        <div className="bg-teal text-white p-4 rounded shadow hover:shadow-lg transition">
          <h3 className="text-xl font-bold">Opportunities</h3>
          <p>Applied: 5</p>
          <p>Ongoing: 2</p>
        </div>
        <div className="bg-yellow p-4 rounded shadow hover:shadow-lg transition">
          <h3 className="text-xl font-bold">Notifications</h3>
          <p>Unread: 3</p>
        </div>
        <div className="bg-red text-white p-4 rounded shadow hover:shadow-lg transition">
          <h3 className="text-xl font-bold">Reports</h3>
          <p>Submitted: 4</p>
        </div>
      </div>
    </div>
  );
};

export default DashboardWidgets;
