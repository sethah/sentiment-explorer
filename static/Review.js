import * as _ from 'lodash'

let red = chroma('#ffbb3a').luminance(0.625)
let blue = chroma('#3affbb').luminance(0.625)
let scaleRed = chroma.scale(['#ddd', red]).mode('lch')
let scaleBlue = chroma.scale(['#ddd', blue]).mode('lch')
let scaleTextRed = chroma.scale(['white', red]).mode('lch')
let scaleTextBlue = chroma.scale(['white', blue]).mode('lch')
let class_labels = ['negative', 'positive']

class Review extends React.Component {
  render() {
    let {
      class_probabilities,
      label,
      lime_tokens,
      lime_scores,
      grem,
    } = this.props

    let winner_index, scaleWinner, scaleLoser
    let winner_name = null
    if (class_probabilities[1] > class_probabilities[0]) {
      winner_index = 1
      scaleWinner = scaleBlue
      scaleLoser = scaleRed
      winner_name = 'pos'
    } else {
      winner_index = 0
      scaleWinner = scaleRed
      scaleLoser = scaleBlue
      winner_name = 'neg'
    }
    let class = winner_name

    let max = Math.max(
      Math.abs(_.min(r.lime_scores)),
      Math.abs(_.max(r.lime_scores))
    )

    function scaleScore(value) {
      if (value < 0) {
        return scaleTextRed(Math.abs(value) / max)
      } else {
        return scaleTextBlue(Math.abs(value) / max)
      }
    }

    let topic_info = null

    return (
      <div
        style={{
          marginBottom: is_review ? 0 : grem,
        }}
      >
          <div style={{ marginTop: grem * 0, marginBottom: grem * 0 }}>
            <div
              style={{
                background: scaleWinner(class_probabilities[winner_index]),
              }}
            >
              <span>
                {winner_name}classification:{' '}
                {class_labels[winner_index]}
              </span>{' '}
              &middot;{' '}
              <span style={{}}>
                {Math.floor(class_probabilities[winner_index] * 1000) / 10}%
                certainty
              </span>
            </div>
              <div
                style={{
                  background: label === 'pos' ? scaleBlue(1) : scaleRed(1),
                }}
              >
                label: {label === 'pos' ? 'positive' : 'negative'} &middot;{' '}
                {label === class ? 'accurate' : 'inaccurate'} classification
              </div>
          </div>

        <div style={{ marginBottom: grem * 0 }}>
          <div>
            <div>
              <div style={{ textIndent: grem * 1 }}>
                {lime_tokens.map((t, i) => {
                  let score = lime_scores[i]
                  let background = analyze
                    ? Math.abs(score) / max > threshold
                      ? scaleScore(score)
                      : 0
                    : 'transparent'
                  return (
                    <span key={t}>
                      {' '}
                      <span
                        style={{
                          backgroundImage: `linear-gradient(${background}, ${background})`,
                          backgroundSize: '4px 5px',
                          backgroundRepeat: 'repeat-x',
                          backgroundPosition: `0 0.85em`,
                          paddingBottom: 2,
                        }}
                      >
                        {t}
                      </span>{' '}
                    </span>
                  )
                })}
              </div>
            </div>
          </div>
        </div>
      </div>
    )
  }
}

export default Review