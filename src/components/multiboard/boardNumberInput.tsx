import { SetNumber } from "../../types";

const BoardNumberInput = ({ setBoardNumber, label }: { setBoardNumber: SetNumber, label: string }) => {
  const handleChange = (e: any) => {
    setBoardNumber(parseInt(e.target.value));
  }

  return (
    <div className="text-white">
      <label className="form-check-label" htmlFor={label}>
        {label}:&nbsp;
      </label>
      <input type="number" id={label} onChange={handleChange} min={1} max={64} />
    </div>
  )
}

export default BoardNumberInput;