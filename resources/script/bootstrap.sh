RUN_NAME="toad_ocr_engine"

if [ -f "./output/bin/${RUN_NAME}" ]; then
    ./output/bin/${RUN_NAME}
else
  echo "./output/bin/${RUN_NAME} not found! please build first"
fi