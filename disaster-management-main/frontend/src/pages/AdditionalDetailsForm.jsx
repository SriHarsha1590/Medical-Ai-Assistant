import React from "react";
import { Formik, Form, Field } from "formik";
import * as Yup from "yup";

const AdditionalDetailsForm = ({ initialValues, onBack, onSubmit }) => {
  const validationSchema = Yup.object({
    fullName: Yup.string().required("Full Name is required"),
    city: Yup.string().required("City is required"),
    area: Yup.string().required("Area is required"),
    profession: Yup.string().required("Profession is required"),
    mobileNumber: Yup.string().required("Mobile Number is required"),
    availabilityDropdown: Yup.string().required("Availability is required"),
  });

  const handleSubmit = (values) => {
    onSubmit(values);
  };

  return (
    <div className="flex items-center justify-center">
      <div className="max-w-md w-full mx-auto bg-white rounded-lg shadow-lg overflow-hidden my-10">
        <Formik
          initialValues={initialValues}
          validationSchema={validationSchema}
          onSubmit={handleSubmit}
        >
          {({ errors, touched }) => (
            <Form className="px-8 py-6">
              <div className="space-y-4">
                {/* Full Name Field */}
                <div>
                  <label htmlFor="fullName" className="block text-gray-700 font-medium mb-1">
                    Full Name
                  </label>
                  <Field
                    id="fullName"
                    name="fullName"
                    type="text"
                    placeholder="Full Name"
                    className="w-full border border-gray-300 rounded-md py-2 px-3 focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                  {errors.fullName && touched.fullName && (
                    <div className="text-red-500 text-sm">{errors.fullName}</div>
                  )}
                </div>

                {/* City Field */}
                <div>
                  <label htmlFor="city" className="block text-gray-700 font-medium mb-1">
                    City
                  </label>
                  <Field
                    id="city"
                    name="city"
                    type="text"
                    placeholder="City"
                    className="w-full border border-gray-300 rounded-md py-2 px-3 focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                  {errors.city && touched.city && (
                    <div className="text-red-500 text-sm">{errors.city}</div>
                  )}
                </div>

                {/* Area Field */}
                <div>
                  <label htmlFor="area" className="block text-gray-700 font-medium mb-1">
                    Area
                  </label>
                  <Field
                    id="area"
                    name="area"
                    type="text"
                    placeholder="Area"
                    className="w-full border border-gray-300 rounded-md py-2 px-3 focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                  {errors.area && touched.area && (
                    <div className="text-red-500 text-sm">{errors.area}</div>
                  )}
                </div>

                {/* Profession Field */}
                <div>
                  <label htmlFor="profession" className="block text-gray-700 font-medium mb-1">
                    Profession
                  </label>
                  <Field
                    id="profession"
                    name="profession"
                    type="text"
                    placeholder="Profession"
                    className="w-full border border-gray-300 rounded-md py-2 px-3 focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                  {errors.profession && touched.profession && (
                    <div className="text-red-500 text-sm">{errors.profession}</div>
                  )}
                </div>

                {/* Mobile Number Field */}
                <div>
                  <label htmlFor="mobileNumber" className="block text-gray-700 font-medium mb-1">
                    Mobile Number
                  </label>
                  <Field
                    id="mobileNumber"
                    name="mobileNumber"
                    type="text"
                    placeholder="Mobile Number"
                    className="w-full border border-gray-300 rounded-md py-2 px-3 focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                  {errors.mobileNumber && touched.mobileNumber && (
                    <div className="text-red-500 text-sm">{errors.mobileNumber}</div>
                  )}
                </div>

                {/* Availability Dropdown */}
                <div>
                  <label htmlFor="availabilityDropdown" className="block text-gray-700 font-medium mb-1">
                    Availability
                  </label>
                  <Field
                    id="availabilityDropdown"
                    name="availabilityDropdown"
                    as="select"
                    className="w-full border border-gray-300 rounded-md py-2 px-3 focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="">Select Availability</option>
                    <option value="full-time">Full Time</option>
                    <option value="part-time">Part Time</option>
                  </Field>
                  {errors.availabilityDropdown && touched.availabilityDropdown && (
                    <div className="text-red-500 text-sm">{errors.availabilityDropdown}</div>
                  )}
                </div>
              </div>

              {/* Buttons */}
              <div className="flex justify-between mt-6">
                <button
                  type="button"
                  onClick={onBack}
                  className="py-2 px-4 bg-gray-300 text-gray-800 rounded-lg focus:outline-none hover:bg-gray-400"
                >
                  Back
                </button>
                <button
                  type="submit"
                  className="py-2 px-4 bg-green-500 text-white rounded-lg focus:outline-none hover:bg-green-600"
                >
                  Submit
                </button>
              </div>
            </Form>
          )}
        </Formik>
      </div>
    </div>
  );
};

export default AdditionalDetailsForm;
