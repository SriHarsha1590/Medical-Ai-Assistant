import React from "react";
import Button from "../Button/Button"; // Import the Button component

const Filter = ({ filters, onFilterChange, onApplyFilters, setIsFilterVisible }) => {
  const handleChange = (e) => {
    const { name, value } = e.target;
    onFilterChange({
      ...filters,
      [name]: value,
    });
  };

  const handleApplyFilters = () => {
    setIsFilterVisible(false); // Close the filter modal after applying filters
  };

  const handleClearFilters = () => {
    onFilterChange({
      type: "All",
      severity: "All",
      status: "All",
    });
  };

  return (
    <form className="space-y-6">
      {/* Filter by Disaster Type */}
      <div>
        <label className="block text-lg font-medium text-navy mb-2">Disaster Type</label>
        <select
          name="type"
          value={filters.type}
          onChange={handleChange}
          className="w-full p-3 border-2 border-gray-300 rounded-lg focus:outline-none focus:border-teal transition bg-white text-black"
        >
          <option value="All">All</option>
          <option value="Earthquake">Earthquake</option>
          <option value="Flood">Flood</option>
          <option value="Hurricane">Hurricane</option>
          <option value="Tornado">Tornado</option>
          <option value="Wildfire">Wildfire</option>
          <option value="Landslide">Landslide</option>
          <option value="Volcanic Eruption">Volcanic Eruption</option>
          <option value="Tsunami">Tsunami</option>
        </select>
      </div>

      {/* Filter by Severity */}
      <div>
        <label className="block text-lg font-medium text-navy mb-2">Severity</label>
        <select
          name="severity"
          value={filters.severity}
          onChange={handleChange}
          className="w-full p-3 border-2 border-gray-300 rounded-lg focus:outline-none focus:border-teal transition bg-white text-black"
        >
          <option value="All">All</option>
          <option value="Severe">Severe</option>
          <option value="Moderate">Moderate</option>
          <option value="Low">Low</option>
          <option value="High">High</option>
        </select>
      </div>

      {/* Filter by Status */}
      <div>
        <label className="block text-lg font-medium text-navy mb-2">Disaster Status</label>
        <select
          name="status"
          value={filters.status}
          onChange={handleChange}
          className="w-full p-3 border-2 border-gray-300 rounded-lg focus:outline-none focus:border-teal transition bg-white text-black"
        >
          <option value="All">All</option>
          <option value="Ongoing">Ongoing</option>
          <option value="Resolved">Resolved</option>
          <option value="Under Recovery">Under Recovery</option>
          <option value="New">New</option>
        </select>
      </div>

      <div className="flex space-x-4">
        {/* Apply Filters Button */}
        <Button
          variant="primary"
          size="large"
          className="w-full mt-6 py-3 font-semibold text-white bg-teal hover:bg-teal-dark focus:outline-none rounded-lg transition"
          onClick={handleApplyFilters} // This should close the modal
        >
          Apply Filters
        </Button>

        {/* Clear Filters Button */}
        <Button
          variant="secondary"
          size="large"
          className="w-full mt-6 py-3 font-semibold text-white bg-gray-500 hover:bg-gray-700 focus:outline-none rounded-lg transition"
          onClick={handleClearFilters}
        >
          Clear Filters
        </Button>
      </div>
    </form>
  );
};

export default Filter;
