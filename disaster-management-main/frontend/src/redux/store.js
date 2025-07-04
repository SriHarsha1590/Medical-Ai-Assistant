import { configureStore, combineReducers } from "@reduxjs/toolkit";
import { persistReducer, persistStore } from "redux-persist";
import storage from "redux-persist/lib/storage";
import userReducer from "./userSlice";

// Combine reducers to include user and volunteer opportunities
const rootReducer = combineReducers({
  user: userReducer,
});

// Configuration for Redux Persist
const persistConfig = {
  key: "root", // Root key for persistence
  storage, // Use localStorage for storing state
  version: 1, // Versioning for migrations
};

// Persist the combined reducer
const persistedReducer = persistReducer(persistConfig, rootReducer);

// Configure the store
export const store = configureStore({
  reducer: persistedReducer,
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({ serializableCheck: false }), // Disable serializable checks for non-serializable actions
});

// Create a persistor to persist the store
export const persistor = persistStore(store);
