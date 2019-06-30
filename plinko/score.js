const outputs = [];

function onScoreUpdate(dropPosition, bounciness, size, bucketLabel) {
  outputs.push([dropPosition, bounciness, size, bucketLabel]);
}

const minMax = (data, featureCount) => {
  const clonedData = _.cloneDeep(data);

  for (let i = 0; i < featureCount; i++) {
    const column = clonedData.map(row => row[i]);
    const min = Math.min(...column);
    const max = Math.max(...column);

    for (let j = 0; j < clonedData.length; j++) {
      clonedData[j][i] = (clonedData[j][i] - min) / (max - min);
    }
  }
  return clonedData;
}

const knn = (data, point, k) => {
  const frequencyMap = data
  .map(row => [
      distance(row.slice(0, -1), point),
      row[row.length -1]
    ]
  )
  .sort((a, b) => {
    return a[0] - b[0]
  })
  .slice(0, k)
  // Make the frequency map
  .reduce((accu, [_, bucket]) => {
    const currentValue = accu[bucket]
    accu[bucket] = currentValue ? currentValue + 1 : 1;
    return accu;
  }, {});
// Sort for most common bucket and return bucket
  return Number(Object.entries(frequencyMap).sort((a, b) => b[1] - a[1])[0][0]);
}

const distance = (pointA, pointB) => {
  // Put identical features in an array together
  return pointA
    .reduce((accu, dataPoint, i) => {
      const newArr = [dataPoint, pointB[i]];
      accu.push(newArr)
      return accu;
    }, [])
    .map(([a, b]) => (a - b)**2)
    .reduce((accu, next) => {
      return accu + next
    }, 0) ** 0.5;
}

const runAnalysis = () => {
  const testSetSize = 50;
  const k = 10;

  const kValues = Array.from(new Array(3), (_, i) => i).forEach(feature => {
    const data = outputs.map(row => [row[feature], row[row.length - 1]])
    const [testSet, trainingSet] = splitDataset(minMax(data, 1), testSetSize);
    const numberCorrect = testSet
    .map((set) => {
      const actualBucket = set[set.length - 1];

      const predictiveBucket = knn(trainingSet, set.slice(0, -1), k);
      return predictiveBucket === actualBucket ? true : false;
    })
    .filter(a => a)
    .length;

    console.log(`Accuracy for ${feature}: ${numberCorrect / testSetSize}`)
  });
}

const splitDataset = (data, testCount) => {
  const shuffled = _.shuffle(data);
  const testSet = shuffled.slice(0, testCount);
  const trainingSet = shuffled.slice(testCount);

  return [testSet, trainingSet];
}
