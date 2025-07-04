import React, { useState } from "react";
import { useSelector } from "react-redux";
import { Link } from "react-router-dom";
import Modal from "react-modal";

Modal.setAppElement("#root"); // Set the root app element for accessibility

const OpportunityCard = ({ opportunity }) => {
  const currentUser = useSelector((state) => state.user.currentUser);
  const role = useSelector((state) => state.user.role);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [modalContent, setModalContent] = useState("");

  const handleSignUpClick = () => {
    if (!currentUser) {
      setModalContent(
        <>
          <p>You must log in as a volunteer to sign up for this opportunity.</p>
          <Link to="/signup" className="text-blue-500 underline mr-4">
            Not a volunteer? Sign up here.
          </Link>
        </>
      );
      setIsModalOpen(true);
    } else if (role === "volunteer") {
      setModalContent("Thank you for signing up! Your request is pending.");
      setIsModalOpen(true);
    } else {
      setModalContent("You must be a volunteer to sign up for this opportunity.");
      setIsModalOpen(true);
    }
  };

  return (
    <div className="bg-gray-light rounded-lg shadow-lg p-4 hover:scale-105 transition">
      <h3 className="text-navy-dark text-xl font-bold mb-2">{opportunity.title}</h3>
      <p className="text-gray mb-2">{opportunity.description}</p>
      
      {opportunity.location?.address && (
        <p className="text-teal font-semibold mb-2">{opportunity.location.address}</p>
      )}
      
      <p className="text-gray-dark">
        <strong>Date:</strong> {opportunity.date ? new Date(opportunity.date).toLocaleDateString() : "Ongoing"}
      </p>
      
      {opportunity.type && (
        <p className="text-gray mb-2">
          <strong>Type & Status:</strong> {opportunity.type} | {opportunity.opportunityStatus || "N/A"}
        </p>
      )}
      
      {opportunity.currentVolunteers !== undefined && opportunity.maxVolunteers && (
        <p className="text-gray mb-2">
          <strong>Volunteers:</strong> {opportunity.currentVolunteers}/{opportunity.maxVolunteers}
        </p>
      )}
      
      {opportunity.deadline && (
        <p className="text-gray mb-2">
          <strong>Deadline:</strong> {new Date(opportunity.deadline).toLocaleDateString()}
        </p>
      )}
      
      {opportunity.resources && opportunity.resources.length > 0 && (
        <div className="mt-4">
          <p className="text-gray mb-2">
            <strong>Resources:</strong>
          </p>
          <ul className="list-disc list-inside text-gray">
            {opportunity.resources.map((resource, index) => (
              <li key={index}>
                <a href={resource} target="_blank" rel="noopener noreferrer" className="text-teal-500 underline">
                  {resource}
                </a>
              </li>
            ))}
          </ul>
        </div>
      )}

      <button
        className="mt-4 bg-teal-500 text-white py-2 px-4 rounded hover:bg-teal-800"
        onClick={handleSignUpClick}
      >
        Sign Up
      </button>

      {/* Modal */}
      <Modal
        isOpen={isModalOpen}
        onRequestClose={() => setIsModalOpen(false)}
        className="bg-white p-6 rounded-lg shadow-lg max-w-md mx-auto"
        overlayClassName="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center"
      >
        <div className="text-center">
          {typeof modalContent === "string" ? (
            <p className="text-navy mb-4">{modalContent}</p>
          ) : (
            modalContent
          )}
          <button
            className="mt-4 bg-teal-400 text-white py-2 px-4 rounded hover:bg-teal-800"
            onClick={() => setIsModalOpen(false)}
          >
            Close
          </button>
        </div>
      </Modal>
    </div>
  );
};

export default OpportunityCard;
