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
    if (!class_probabilities) {
      return null
    }
    let threshold = 0.5
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
    let clazz = winner_name

    let max = Math.max(
      Math.abs(_.min(lime_scores)),
      Math.abs(_.max(lime_scores))
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
          marginBottom: grem,
        }}
      >
        <div style={{ marginTop: grem * 0, marginBottom: grem * 0 }}>
          <div
            style={{
              background: scaleWinner(class_probabilities[winner_index]),
            }}
          >
            <span>
              classification:{' '}
              {class_labels[winner_index]}
            </span>{' '}
            &middot;{' '}
            <span style={{}}>
              {Math.floor(class_probabilities[winner_index] * 1000) / 10}%
              certainty
              </span>
          </div>
        </div>

        <div style={{ marginBottom: grem * 0 }}>
          <div>
            <div>
              <div style={{ textIndent: grem * 1 }}>
                {lime_tokens.map((t, i) => {
                  let score = lime_scores[i]
                  let background = Math.abs(score) / max > threshold ? scaleScore(score) : 0
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

const Wrapper = styled.div`
  color: #232323;
  border-top: 4px solid #fcb431;
  max-width: 940px;
  flex-grow: 1;
  margin: 2rem;
  background: #fff;
  padding: 2rem;
  font-size: 1em;
  box-shadow: 1px 1px 3px rgba(0,0,0,0.3);
  font-size: 1em;
  line-height: 1.4;

  @media(max-width: 500px) {
    margin: 0;
  }
`

const Title = styled.h1`
  font-size: 1.5em;
  margin: 0 0 1rem;
  display: flex;
  align-items: center;
  flex-wrap: wrap;

  @media(max-width: 500px) {
    justify-content: center;
  }
`

const AppName = styled.span`
  background: #2085bc;
  font-weight: 200;
  color: #fff;
  padding: 0.5rem 1rem;
  line-height: 1;
  border-radius: 2rem;
  margin-left: auto;

  @media(max-width: 500px) {
    margin: 1rem 0 0;
  }
`

const OutputText = styled.p`
    font-size: 0.5em;
    color: ${props => props.inputColor || "palevioletred"};
`;

const SentimentOutput = ({ sentiment }) => {
}

const LinkHome = styled.a`
  background-image: url('static/cloudera.png');
  background-size: 300px 40px;
  backround-repeat: no-repeat;
  width: 300px;
  height: 40px;
  display: block;
  margin: 0 1rem 0 0;

  @media(max-width: 500px) {
    background-image: url('static/ai2-logo-header-crop.png');
    background-size: 89px 71px;
    width: 89px;
    height: 71px;
  }
`

const Intro = styled.div`
  margin: 2em 0;

  @media(max-width: 500px) {
    font-size: 0.8em;
  }
`

const TextInputWrapper = styled.div`
  position: relative;
`

const Loading = styled.div`
  position: absolute;
  bottom: 1rem;
  right: 1rem;
  display: flex;
  align-items: center;
  font-size: 0.8em;
  color: #8c9296;
`

const Error = styled(Loading)`
  color: red;
`

const LoadingText = styled.div`
  padding-left: 0.5rem;
`

const InputOutput = styled.div`
  display: flex;

  @media(max-width: 500px) {
    display: block;
  }
`

const InputOutputColumn = styled.div`
  flex: 1 1 50%;

  :first-child {
    padding-right: 1rem;
  }

  :last-child {
    padding-left: 1rem;
  }

  @media(max-width: 500px) {
    :first-child,
    :last-child {
      padding: 0;
    }

    :first-child {
      padding: 0 0 1rem;
    }
  }
`

const TextInput = styled.textarea`
  display: block;
  width: 100%;
  font-size: 1.25em;
  min-height: 100px;
  border: 1px solid rgba(0, 0, 0, 0.2);
  box-shadow: inset 1px 1px 4px rgba(0, 0, 0, 0.1);
  padding: 1rem;
  border-radius: 0.25rem;
`

const Button = styled.button`
  color: #fff!important;
  background: #2085bc;
`

const ListItem = styled.li`
  margin: 0 0 0.5rem;
`

const InputHeader = styled.h2`
  font-weight: 600;
  font-size: 1.1em;
  margin: 0 0 1rem;
  padding: 0 0 0.5rem;
  border-bottom: 1px solid #eee;
`

const ChoiceList = styled.ul`
  padding: 0;
  margin: 0;
  flex-wrap: wrap;
  list-style-type: none;
`

const ChoiceItem = styled.button`
  color: #2085bc;
  cursor: pointer;
  background: transparent;
  display: inline-flex;
  align-items: center;
  line-height: 1;
  font-size: 1.15em;
  border: none;
  border-bottom: 2px solid transparent;
`

const UndoButton = styled(ChoiceItem)`
  color: #8c9296;
`

const Probability = styled.span`
  color: #8c9296;
  margin-right: 0.5rem;
  font-size: 0.8em;
  min-width: 4em;
  text-align: right;
`

const Token = styled.span`
  font-weight: 600;
`

const OutputSentence = styled.div`
  margin: 20px;
  font-family: monospace;
  flex: 1;
`

const OutputToken = styled.span`
  cursor: pointer;

  :hover {
      font-weight: bold;
  }
`

const OutputSpace = styled.span``

const ModelChoice = styled.span`
  font-weight: ${props => props.selected ? 'bold' : 'normal'};
  color: ${props => props.selected ? 'black' : 'lightgray'};
  cursor: ${props => props.selected ? 'default' : 'pointer'};
`

const Footer = styled.div`
  margin: 2rem 0 0 0;
`

const DEFAULT = "Avengers Endgame was without a doubt the neatest movie ever.";

function addToUrl(output, choice) {
  if ('history' in window) {
    window.history.pushState(null, null, '?text=' + encodeURIComponent(output + (choice || '')))
  }
}

function loadFromUrl() {
  const params =
    document.location.search.substr(1).split('&').map(p => p.split('='));
  const text = params.find(p => p[0] === 'text');
  return Array.isArray(text) && text.length == 2 ? decodeURIComponent(text.pop()) : null;
}

function trimRight(str) {
  return str.replace(/ +$/, '');
}

const DEFAULT_MODEL = "NBSVM"

class App extends React.Component {

  constructor(props) {
    super(props)

    this.currentRequestId = 0;

    this.state = {
      output: loadFromUrl() || DEFAULT,
      prev: null,
      words: null,
      label: null,
      lime_tokens: null,
      lime_scores: null,
      class_probabilities: null,
      loading: false,
      error: false,
      model: DEFAULT_MODEL,
      sentiment: "neg"
    }

    this.switchModel = this.switchModel.bind(this)
    this.choose = this.choose.bind(this)
    this.debouncedChoose = _.debounce(this.choose, 1000)
    this.setOutput = this.setOutput.bind(this)
    this.runOnEnter = this.runOnEnter.bind(this)
  }

  setOutput(evt) {
    const value = evt.target.value
    const trimmed = trimRight(value);

    this.setState({
      prev: this.state.output,
      output: value,
      label: null,
      loading: trimmed.length > 0
    })
    this.debouncedChoose()
  }

  createRequestId() {
    const nextReqId = this.currentRequestId + 1;
    this.currentRequestId = nextReqId;
    return nextReqId;
  }

  componentDidMount() {
    this.choose()
    if ('history' in window) {
      window.addEventListener('popstate', () => {
        const fullText = loadFromUrl();
        const doNotChangeUrl = fullText ? true : false;
        const output = fullText || DEFAULT;
        this.setState({
          output,
          loading: true,
          words: null,
          label: null,
          lime_tokens: null,
          lime_scores: null,
          class_probabilities: null,
          model: this.state.model
        }, () => this.choose(undefined, doNotChangeUrl));
      })
    }
  }

  switchModel() {
    const newModel = this.state.model == "BERT" ? "NBSVM" : "BERT"
    this.setState({ loading: true, error: false, model: newModel }, this.choose)
  }

  choose(choice = undefined, doNotChangeUrl) {
    // reset the state here or you'll just layer new stuff on top of it
    this.setState({
      words: null,
      label: null,
      lime_tokens: null,
      lime_scores: null,
      class_probabilities: null,
      loading: true,
      error: false
    })

    // strip trailing spaces
    const trimmedOutput = trimRight(this.state.output);
    if (trimmedOutput.length === 0) {
      this.setState({ loading: false });
      return;
    }

    const payload = {
      previous: trimmedOutput,
      numsteps: 5,
      model_name: this.state.model
    }

    const currentReqId = this.createRequestId();
    const endpoint = '/predict'

    if ('history' in window && !doNotChangeUrl) {
      addToUrl(this.state.output, choice);
    }
    gtag('config', window.googleUA, {
      page_location: document.location.toString()
    });

    fetch(endpoint, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload)
    })
      .then(response => response.json())
      .then(data => {
        if (this.currentRequestId === currentReqId) {
          // If the user entered text by typing don't overwrite it, as that feels
          // weird. If they clicked it overwrite it
          const output = choice === undefined ? this.state.output : data.output
          this.setState({ ...data, output, loading: false })
        }
      })
      .catch(err => {
        console.error('Error trying to communicate with the API:', err);
        this.setState({ error: true, loading: false });
      });
  }

  // Temporarily (?) disabled
  runOnEnter(e) {
    if (e.key === 'Enter') {
      e.preventDefault()
      e.stopPropagation()
      this.choose()
    }
  }

  render() {
    return (
      <Wrapper>
        <Title>
          <LinkHome href="https://experiments.fastforwardlabs.com" target="_blank"></LinkHome>
          <AppName>Textflix Sentiment Analyzer</AppName>
        </Title>
        <Intro>
          This is a sentiment analyzer. You write a review and we tell you how you felt about it.
        </Intro>
        <span>Choose which model to use:
        <ModelSwitcher model={this.state.model} switchModel={this.switchModel} />
        </span>
        <InputOutput>
          <InputOutputColumn>
            <InputHeader>Sentence:</InputHeader>
            <TextInputWrapper>
              <TextInput type="text"
                value={this.state.output}
                onChange={this.setOutput} />
              {this.state.loading ? (
                <Loading>
                  <img src="/static/loading-bars.svg" width="25" height="25" />
                  <LoadingText>Loading</LoadingText>
                </Loading>
              ) : null}
              {this.state.error ? (
                <Error>
                  ⚠️ Something went wrong. Please try again.
                </Error>
              ) : null}
            </TextInputWrapper>
          </InputOutputColumn>
          <InputOutputColumn>
            <InputHeader>Sentiment:</InputHeader>
            <Review class_probabilities={this.state.class_probabilities}
              label={this.state.label}
              lime_tokens={this.state.lime_tokens}
              lime_scores={this.state.lime_scores}
              grem='1'
            />
          </InputOutputColumn>
        </InputOutput>
        <Footer>
          Built at <a href="https://experiments.fastforwardlabs.com/" target="_blank">Cloudera Fast Forward Labs</a>
        </Footer>
      </Wrapper>
    )
  }
}

const ModelSwitcher = ({ model, switchModel }) => (
  <span className="model-switcher" onClick={switchModel}>
    {' '}
    <ModelChoice selected={model === "BERT"}>BERT</ModelChoice>
    {' '}
    <ModelChoice selected={model === "NBSVM"}>NBSVM</ModelChoice>
    {' '}
  </span>
)

const formatProbability = prob => {
  prob = prob * 100
  return `${prob.toFixed(1)}%`
}

const Sentences = ({ sentences }) => {
  if (!sentences) { return null }

  return <p>{sentences.join(" ")}</p>
}

ReactDOM.render(<App />, document.getElementById("app"))
