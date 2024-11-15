interface Study {
  id: string,
  name: string
}

interface ModelRefs {
  piecesModelRef: any,
  xcornersModelRef: any
}

interface MovesData {
  sans: string[],
  from: number[],
  to: number[],
  targets: number[]
}
interface MovesPair {
  "move1": MovesData,
  "move2": MovesData | null,
  "moves": MovesData | null
}

type CornersKey = "h1" | "a1" | "a8" | "h8";
interface CornersPayload {
  key: CornersKey,
  xy: number[]
}
type CornersDict = {[key in CornersKey]: number[]};

interface Game {
  fen: string,
  moves: string,
  start: string,
  lastMove: string,
  greedy: boolean
}

interface User {
  token: string,
  username: string
}

interface RootState {
  game1: Game,
  game2: Game,
  corners: CornersDict,
  cornersMultiple: { [board: string]: CornersDict },
  user: User,
  gameMultiple: { board1: Game, board2: Game } // Add this line
}

interface TwoBoardsState {
  game1: Game,
  game2: Game,
  corners1: CornersDict,
  corners2: CornersDict
}

interface TwoBoardsPayload {
  key: CornersKey,
  xy1: number[],
  xy2: number[]
}

type Mode = "record" | "upload" | "broadcast" | "play" | "multiboard";

type SetBoolean = React.Dispatch<React.SetStateAction<boolean>>
type SetString = React.Dispatch<React.SetStateAction<string>>
type SetStringArray = React.Dispatch<React.SetStateAction<string[]>>
type SetNumber = React.Dispatch<React.SetStateAction<number>>
type SetStudy = React.Dispatch<React.SetStateAction<Study | null>>

export type {
  RootState, Study, ModelRefs, MovesData, MovesPair,
  CornersDict, CornersKey, CornersPayload, Game, TwoBoardsState, TwoBoardsPayload,
  SetBoolean, SetString, SetStringArray, SetNumber, Mode,
  SetStudy
}