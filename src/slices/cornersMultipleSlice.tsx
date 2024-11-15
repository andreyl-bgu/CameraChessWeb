import { createSlice } from '@reduxjs/toolkit';
import { CornersDict, CornersPayload, RootState } from "../typesMultiple";
import { useSelector } from 'react-redux';

const initialState: { [board: string]: CornersDict } = {
  board1: { "h1": [50, -100], "a1": [0, -100], "a8": [0, -150], "h8": [50, -150] },
  board2: { "h1": [50, -100], "a1": [0, -100], "a8": [0, -150], "h8": [50, -150] }
};

interface Action {
  payload: { board: string, data: CornersPayload },
  type: string
}

const cornerMultipleSlice = createSlice({
  name: 'cornerMultiple',
  initialState,
  reducers: {
    cornersSet(state, action: Action) {
      state[action.payload.board][action.payload.data.key] = action.payload.data.xy;
    },
    cornersReset(state, action: { payload: { board: string } }) {
      state[action.payload.board] = initialState[action.payload.board];
    },
    cornersResetAll() {
      return initialState;
    }
  }
});

export const cornerMultipleSelect = (board: string) => {
  return useSelector((state: RootState) => state.cornersMultiple[board]);
};

export const { cornersSet, cornersReset, cornersResetAll } = cornerMultipleSlice.actions;
export default cornerMultipleSlice.reducer;