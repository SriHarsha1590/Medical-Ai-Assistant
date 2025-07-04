import React, { useState, useEffect } from "react";
import Button from "../Button/Button";
import { MdClose } from "react-icons/md"; // Close icon

const Filter = ({
  selectedType,
  filters,
  setFilters,
  closeModal,
  setLoadingError,
  userLocation,
}) => {
  const [selectedFilters, setSelectedFilters] = useState(filters);

  useEffect(() => {
    setSelectedFilters(filters);
  }, [filters]);

  const handleApplyFilters = () => {
    setFilters(selectedFilters);
    closeModal(); // Close the modal after applying filters
  };

  const handleResetFilters = () => {
    setSelectedFilters({
      availability: "all",
      type: "all",
      nearMe: false,
      facilities: [],
      services: [],
      emergencyBeds: "all",
      availabilityStatus: "all",
    });
  };

  const handleCloseModal = () => {
    closeModal(); // Close the modal when the cross mark is clicked
  };

  return (
    <div className="fixed inset-0 bg-gray-800 bg-opacity-50 flex justify-center items-center z-50">
      <div className="bg-white rounded-lg p-6 w-96 relative shadow-lg">
        {/* Close button */}
        <button
          onClick={handleCloseModal}
          className="absolute top-4 right-4 text-gray-600 hover:text-gray-900"
          aria-label="Close Filter"
        >
          <MdClose size={24} />
        </button>

        <h2 className="text-2xl font-bold mb-4">Filters</h2>

        {/* Shelter-specific filters */}
        {selectedType === "shelter" && (
          <>
            <div className="mb-4">
              <label className="block mb-2 font-medium text-gray-700">Type</label>
              <select
                value={selectedFilters.type}
                onChange={(e) =>
                  setSelectedFilters({ ...selectedFilters, type: e.target.value })
                }
                className="w-full border rounded-md p-2 focus:outline-indigo-500"
              >
                <option value="all">All</option>
                <option value="general">General</option>
                <option value="medical">Medical</option>
                <option value="family">Family</option>
                <option value="pet-friendly">Pet-friendly</option>
                <option value="temporary">Temporary</option>
              </select>
            </div>

            <div className="mb-4">
              <label className="block mb-2 font-medium text-gray-700">
                Availability
              </label>
              <select
                value={selectedFilters.availability}
                onChange={(e) =>
                  setSelectedFilters({ ...selectedFilters, availability: e.target.value })
                }
                className="w-full border rounded-md p-2 focus:outline-indigo-500"
              >
                <option value="all">All</option>
                <option value="available">Available</option>
              </select>
            </div>

            <div className="mb-4">
              <label className="flex items-center font-medium text-gray-700">
                <input
                  type="checkbox"
                  checked={selectedFilters.nearMe}
                  onChange={(e) =>
                    setSelectedFilters({ ...selectedFilters, nearMe: e.target.checked })
                  }
                  className="mr-2"
                />
                Near Me
              </label>
            </div>

            <div className="mb-4">
              <label className="block mb-2 font-medium text-gray-700">Facilities</label>
              <select
                multiple
                value={selectedFilters.facilities}
                onChange={(e) =>
                  setSelectedFilters({
                    ...selectedFilters,
                    facilities: Array.from(e.target.selectedOptions, (o) => o.value),
                  })
                }
                className="w-full border rounded-md p-2 focus:outline-indigo-500 h-32"
              >
                <option value="beds">Beds</option>
                <option value="food">Food</option>
                <option value="water">Water</option>
                <option value="medical">Medical</option>
              </select>
            </div>

            <div className="mb-4">
              <label className="block mb-2 font-medium text-gray-700">Services</label>
              <select
                multiple
                value={selectedFilters.services}
                onChange={(e) =>
                  setSelectedFilters({
                    ...selectedFilters,
                    services: Array.from(e.target.selectedOptions, (o) => o.value),
                  })
                }
                className="w-full border rounded-md p-2 focus:outline-indigo-500 h-32"
              >
                <option value="first aid">First Aid</option>
                <option value="transport">Transport</option>
                <option value="shelter assistance">Shelter Assistance</option>
              </select>
            </div>
          </>
        )}

        {/* Hospital-specific filters */}
        {selectedType === "hospital" && (
          <>
            <div className="mb-4">
              <label className="block mb-2 font-medium text-gray-700">
                Availability Status
              </label>
              <select
                value={selectedFilters.availabilityStatus}
                onChange={(e) =>
                  setSelectedFilters({
                    ...selectedFilters,
                    availabilityStatus: e.target.value,
                  })
                }
                className="w-full border rounded-md p-2 focus:outline-indigo-500"
              >
                <option value="all">All</option>
                <option value="Open">Open</option>
                <option value="Full">Full</option>
                <option value="Closed">Closed</option>
              </select>
            </div>

            <div className="mb-4">
              <label className="block mb-2 font-medium text-gray-700">
                Emergency Beds
              </label>
              <select
                value={selectedFilters.emergencyBeds}
                onChange={(e) =>
                  setSelectedFilters({
                    ...selectedFilters,
                    emergencyBeds: e.target.value,
                  })
                }
                className="w-full border rounded-md p-2 focus:outline-indigo-500"
              >
                <option value="all">All</option>
                <option value="0">0</option>
                <option value="1">1+</option>
                <option value="2">2+</option>
              </select>
            </div>
          </>
        )}

        <div className="flex justify-between mt-4">
          <Button
            variant="secondary"
            size="medium"
            onClick={handleResetFilters}
            className="hover:bg-gray-200 hover:text-gray-900 transition duration-300 ease-in-out"
          >
            Reset
          </Button>
          <Button
            variant="primary"
            size="medium"
            onClick={handleApplyFilters}
            className="hover:bg-indigo-600 hover:text-white transition duration-300 ease-in-out"
          >
            Apply Filters
          </Button>
        </div>
      </div>
    </div>
  );
};

export default Filter;
