import { createSlice } from "@reduxjs/toolkit";

// Initial state for user slice
const initialState = {
  currentUser: null, // User data
  token: null, // JWT token
  error: null,
  loading: false,
  role: "none", // Default role
};

const userSlice = createSlice({
  name: "user",
  initialState,
  reducers: {
    signInStart: (state) => {
      state.loading = true;
      state.error = null;
    },
    signInSuccess: (state, action) => {
      state.loading = false;
      state.error = null;
      state.currentUser = action.payload.user;
      state.token = action.payload.token;
      state.role = action.payload.user.role;
    },
    signInFailure: (state, action) => {
      state.loading = false;
      state.error = action.payload;
    },
    updateStart: (state) => {
      state.loading = true;
      state.error = null;
    },
    updateSuccess: (state, action) => {
      state.currentUser = { ...state.currentUser, ...action.payload };
      state.loading = false;
      state.error = null;
    },
    updateFailure: (state, action) => {
      state.loading = false;
      state.error = action.payload;
    },
    signoutSuccess: (state) => {
      // Reset state completely
      Object.assign(state, initialState);
    },
    updateRoleSuccess: (state, action) => {
      if (state.currentUser) {
        state.currentUser.role = action.payload;
      }
      state.role = action.payload;
    },
  },
});

export const {
  signInStart,
  signInSuccess,
  signInFailure,
  updateStart,
  updateSuccess,
  updateFailure,
  signoutSuccess,
  updateRoleSuccess,
} = userSlice.actions;

export default userSlice.reducer;

