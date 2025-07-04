import React, { useEffect, useState } from "react";
import Header from "../components/VolunteerOpportunities/Header";
import OpportunityCard from "../components/VolunteerOpportunities/OpportunityCard";
import FiltersModal from "../components/VolunteerOpportunities/FiltersModal";
import VolunteerMap from "../components/VolunteerOpportunities/VolunteerMap";
import FAQAccordion from "../components/VolunteerOpportunities/FAQAccordion";
import axios from "axios";

const VolunteerPage = () => {
  const [opportunities, setOpportunities] = useState([]);
  const [filteredOpportunities, setFilteredOpportunities] = useState([]);
  const [isFilterModalOpen, setFilterModalOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");

  // Fetch opportunities from backend
  useEffect(() => {
    const fetchOpportunities = async () => {
      try {
        const response = await axios.get("/api/volunteer-opportunities");
        setOpportunities(response.data);
        setFilteredOpportunities(response.data);
      } catch (error) {
        console.error("Error fetching volunteer opportunities", error);
      }
    };
    fetchOpportunities();
  }, []);

  // Filter opportunities based on search query
  useEffect(() => {
    const filtered = opportunities.filter((opportunity) => {
      const query = searchQuery.toLowerCase();
      return (
        opportunity.title.toLowerCase().includes(query) ||
        opportunity.location.address.toLowerCase().includes(query)
      );
    });
    setFilteredOpportunities(filtered);
  }, [searchQuery, opportunities]);

  // Apply filters (from the modal)
  const applyFilters = (filters) => {
    const { location, date, type } = filters;
    const filtered = opportunities.filter((opportunity) => {
      if (location && opportunity.location.address.indexOf(location) === -1) return false;
      if (date && new Date(opportunity.date).toDateString() !== new Date(date).toDateString())
        return false;
      if (type && !opportunity.title.toLowerCase().includes(type.toLowerCase())) return false;
      return true;
    });
    setFilteredOpportunities(filtered);
    setFilterModalOpen(false);
  };

  return (
    <div className="bg-gray-light min-h-screen">
      {/* Header */}
      <div className="p-6 mb-6">
        <Header />
      </div>

      {/* Search Bar and Filter Button */}
      <div className="flex justify-between items-center px-4 py-6 mb-6">
        <input
          type="text"
          placeholder="Search opportunities by title or location"
          className="border border-gray rounded-lg p-2 w-3/4"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
        />
        <button
          className="ml-4 bg-teal hover:bg-teal-800 text-white py-2 px-4 rounded transition duration-300"
          onClick={() => setFilterModalOpen(true)}
        >
          Filter Opportunities
        </button>
      </div>

      {/* Volunteer Opportunities */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 p-6">
        {filteredOpportunities.length > 0 ? (
          filteredOpportunities.map((opportunity) => (
            <OpportunityCard key={opportunity._id} opportunity={opportunity} />
          ))
        ) : (
          <p className="text-center text-gray-dark col-span-full">
            No opportunities found. Try adjusting your search or filters.
          </p>
        )}
      </div>

      {/* Map */}
      <div className="p-6">
        <VolunteerMap opportunities={filteredOpportunities} />
      </div>

      {/* FAQ Section */}
      <div className="p-6">
        <FAQAccordion />
      </div>

      {/* Filters Modal */}
      {isFilterModalOpen && (
        <FiltersModal
          closeModal={() => setFilterModalOpen(false)}
          applyFilters={applyFilters}
        />
      )}
    </div>
  );
};

export default VolunteerPage;
