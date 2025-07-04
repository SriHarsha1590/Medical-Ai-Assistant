import React, { useState } from "react";
import { FaChevronDown, FaChevronUp } from "react-icons/fa"; // Import icons

const FAQAccordion = () => {
  const [open, setOpen] = useState(null);

  const toggle = (index) => setOpen(open === index ? null : index);

  const faqs = [
    { question: "Who can volunteer?", answer: "Anyone above the age of 18." },
    { question: "What is the time commitment?", answer: "Flexible schedules are available." },
    { question: "Do I need experience?", answer: "No prior experience is required." },
  ];

  return (
    <div className="p-6 max-w-xl mx-auto">
      <h2 className="text-2xl font-bold text-center text-navy mb-6">FAQ</h2>
      {faqs.map((faq, index) => (
        <div
          key={index}
          className="mb-4 border border-gray-light rounded-lg p-4 shadow-sm"
        >
          <div
            className="flex justify-between items-center cursor-pointer text-navy font-semibold"
            onClick={() => toggle(index)}
          >
            <span>{faq.question}</span>
            <span>
              {open === index ? (
                <FaChevronUp className="text-teal" />
              ) : (
                <FaChevronDown className="text-gray" />
              )}
            </span>
          </div>
          {open === index && (
            <p className="text-gray mt-2">{faq.answer}</p>
          )}
        </div>
      ))}
    </div>
  );
};

export default FAQAccordion;
