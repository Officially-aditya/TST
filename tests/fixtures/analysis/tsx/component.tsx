interface Props {
  title: string;
}

export const Card = ({ title }: Props) => {
  return <article>{title}</article>;
};
