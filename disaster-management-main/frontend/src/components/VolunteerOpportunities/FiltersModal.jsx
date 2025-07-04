import React, { useState } from "react";

const FiltersModal = ({ closeModal, applyFilters }) => {
  const [filters, setFilters] = useState({ location: "", date: "", type: "" });

  const handleChange = (e) =>
    setFilters({ ...filters, [e.target.name]: e.target.value });

  const handleApply = () => applyFilters(filters);

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white p-6 rounded-lg shadow-lg">
        <h2 className="text-navy text-xl font-bold mb-4">Filter Opportunities</h2>
        <input
          type="text"
          name="location"
          placeholder="Location"
          className="border border-gray rounded-lg p-2 w-full mb-4"
          value={filters.location}
          onChange={handleChange}
        />
        <input
          type="date"
          name="date"
          className="border border-gray rounded-lg p-2 w-full mb-4"
          value={filters.date}
          onChange={handleChange}
        />
        <input
          type="text"
          name="type"
          placeholder="Type (e.g., awareness, shelter)"
          className="border border-gray rounded-lg p-2 w-full mb-4"
          value={filters.type}
          onChange={handleChange}
        />
        <div className="flex justify-between">
          <button
            className="bg-teal text-white px-4 py-2 rounded hover:bg-teal-light"
            onClick={handleApply}
          >
            Apply Filters
          </button>
          <button
            className="bg-red text-white px-4 py-2 rounded"
            onClick={closeModal}
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};

export default FiltersModal;
