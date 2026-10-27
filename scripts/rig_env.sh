# Source after changing to the repository root.
if [[ -f .env ]]; then
    set -a
    source .env
    set +a
fi
: "${EMBODIMENT:?Set EMBODIMENT in .env or your environment}"
case "${EMBODIMENT}" in
    robotiq|umi|sharpa) ;;
    *) echo "Unsupported Franka embodiment: ${EMBODIMENT}" >&2; exit 2 ;;
esac
