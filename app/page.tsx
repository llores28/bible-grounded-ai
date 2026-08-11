const budget = [
  ["RTX 5090 development workstation", "$6,500", "GPU, 128 GB RAM, fast storage, power and cooling"],
  ["Cloud compute reserve", "$2,000", "Benchmarking and training runs beyond local capacity"],
  ["Data, legal and accounting", "$1,500", "Licensing review and responsible administration"],
  ["Scholar and pastoral review", "$3,000", "Hebrew, Greek, theology and moral-case evaluation"],
  ["Outreach and media", "$500", "Domain, campaign materials and project communications"],
  ["Fees and contingency", "$1,500", "Processing fees, price changes and unexpected costs"],
];

const phases = [
  ["01", "Govern", "Publish the moral constitution, licensing register, test methodology and project boundaries."],
  ["02", "Establish a baseline", "Measure the pinned Apertus 8B text model as the primary interpreter; use smaller checkpoints only for routing or retrieval if they pass their narrower gates."],
  ["03", "Train carefully", "Run memory-efficient Hebrew, Aramaic, Greek and Latin experiments on one RTX 5090."],
  ["04", "Ground every claim", "Connect exact Scripture and manuscript transcriptions through citation-verified retrieval."],
  ["05", "Review openly", "Invite qualified reviewers, publish failures and report what the evidence actually supports."],
];

const principles = [
  "Source before assertion",
  "Interpretation clearly labeled",
  "Human dignity protected",
  "Uncertainty stated plainly",
  "Independent review welcomed",
  "Results and failures published",
];

export default function Home() {
  return (
    <main>
      <header className="site-header">
        <a className="brand" href="#top" aria-label="Bible-Grounded AI home">
          <span className="brand-mark" aria-hidden="true"><i /><i /><i /></span>
          <span>Bible-Grounded <b>AI</b></span>
        </a>
        <nav aria-label="Primary navigation">
          <a href="#mission">Mission</a>
          <a href="#method">Method</a>
          <a href="#roadmap">Roadmap</a>
          <a href="#faq">FAQ</a>
        </nav>
        <a className="button button-small" href="#support">Support the project</a>
      </header>

      <section className="hero" id="top">
        <div className="hero-glow glow-one" />
        <div className="hero-glow glow-two" />
        <div className="hero-copy">
          <p className="eyebrow"><span /> An open research initiative</p>
          <h1>Ancient wisdom.<br /><em>Accountable AI.</em></h1>
          <p className="hero-lede">
            We are building a transparent AI research system that studies Scripture in its biblical languages and applies clearly identified biblical principles to everyday life—without claiming to speak for God.
          </p>
          <div className="hero-actions">
            <a className="button" href="#support">Support the proof of concept <span>→</span></a>
            <a className="text-link" href="#method">Explore the method <span>↓</span></a>
          </div>
          <div className="trust-line">
            <span>Open methods</span><span>Verified citations</span><span>Human review</span>
          </div>
        </div>

        <div className="hero-visual" aria-label="Biblical languages flowing into a transparent, reviewed AI system">
          <div className="orbit orbit-one" />
          <div className="orbit orbit-two" />
          <div className="language language-hebrew"><strong>בְּרֵאשִׁית</strong><small>Biblical Hebrew</small></div>
          <div className="language language-greek"><strong>Ἐν ἀρχῇ</strong><small>Koine Greek</small></div>
          <div className="language language-latin"><strong>In principio</strong><small>Latin</small></div>
          <div className="core">
            <span className="core-cross">✦</span>
            <strong>Scripture</strong>
            <small>Evidence · Context · Reasoning</small>
          </div>
          <div className="status-card">
            <span className="status-dot" />
            <div><small>Current stage</small><strong>Policy core built · data review next</strong></div>
          </div>
        </div>
      </section>

      <section className="manifesto" id="mission">
        <p className="section-label">Why this matters</p>
        <div className="manifesto-grid">
          <h2>Convincing words are not the same as <em>trustworthy guidance.</em></h2>
          <div>
            <p>Many AI systems can sound authoritative while blending an exact biblical text, a translation choice, a theological interpretation and practical advice.</p>
            <p>This project starts with traceability. People should be able to see what supports an answer, where uncertainty remains, and where faithful Christian traditions differ.</p>
          </div>
        </div>
        <div className="not-claims">
          <span>It will not claim to be</span>
          <b>conscious</b><i>·</i><b>divinely inspired</b><i>·</i><b>a replacement for Scripture, prayer, pastors or professionals</b>
        </div>
      </section>

      <section className="method" id="method">
        <div className="section-heading">
          <div><p className="section-label light">The research method</p><h2>Built to show its work.</h2></div>
          <p>The model is only one component. Reliable guidance requires governed data, exact-source retrieval, explicit reasoning rules and qualified human evaluation.</p>
        </div>
        <div className="method-grid">
          {phases.map(([num, title, text]) => (
            <article className="method-card" key={num}>
              <span>{num}</span><h3>{title}</h3><p>{text}</p>
            </article>
          ))}
        </div>
        <div className="principles">
          {principles.map((item) => <span key={item}><i>✓</i>{item}</span>)}
        </div>
      </section>

      <section className="roadmap" id="roadmap">
        <div className="section-heading dark">
          <div><p className="section-label">Campaign one</p><h2>Fund the first<br /><em>verifiable milestone.</em></h2></div>
          <p>The first campaign funds a reproducible proof of concept—not a claim that a finished “moral AI” can be purchased for $15,000.</p>
        </div>

        <div className="funding-panel">
          <div className="goal-card">
            <p>Initial campaign goal</p>
            <strong>$15,000</strong>
            <div className="goal-track"><i /></div>
            <div className="goal-meta"><span>Campaign preparing to launch</span><b>0%</b></div>
            <a className="button full" href="#support">Join the founding circle <span>→</span></a>
            <small>Campaign platform and tax status will be published before contributions are accepted.</small>
          </div>
          <div className="milestones">
            <div><b>$3K</b><span>Governance, license register and baseline tests</span></div>
            <div><b>$8K</b><span>Development workstation and reproducible baseline</span></div>
            <div><b>$12K</b><span>First language adaptation and measured results</span></div>
            <div><b>$15K</b><span>Independent expert review and public report</span></div>
          </div>
        </div>

        <div className="budget-wrap">
          <div className="budget-head"><h3>Where the funding goes</h3><span>Proposed campaign-one budget</span></div>
          <div className="budget-list">
            {budget.map(([name, amount, detail]) => (
              <div className="budget-row" key={name}><div><b>{name}</b><small>{detail}</small></div><strong>{amount}</strong></div>
            ))}
            <div className="budget-total"><b>Total</b><strong>$15,000</strong></div>
          </div>
        </div>
      </section>

      <section className="commitments">
        <p className="section-label">Public commitments</p>
        <h2>Trust must be <em>earned in public.</em></h2>
        <div className="commitment-grid">
          <article><span>01</span><h3>Legal, traceable data</h3><p>Use public-domain, open-license or properly licensed material with dataset-level provenance.</p></article>
          <article><span>02</span><h3>Honest interpretation</h3><p>Separate explicit text, multi-passage biblical synthesis, historical interpretation and factual organizational alignment.</p></article>
          <article><span>03</span><h3>Verified source claims</h3><p>Require retrieval verification for Scripture, manuscript and original-language claims.</p></article>
          <article><span>04</span><h3>Published limitations</h3><p>Report failed tests, known weaknesses and safety boundaries—not only successful demonstrations.</p></article>
        </div>
      </section>

      <section className="faq" id="faq">
        <div><p className="section-label">Common questions</p><h2>Clear answers.<br /><em>No grand claims.</em></h2></div>
        <div className="faq-list">
          <details open><summary>Is this AI supposed to speak for God?<span>+</span></summary><p>No. It is a human-designed research system that applies a documented biblical framework. It will remain capable of error and its outputs must be verified.</p></details>
          <details><summary>Why not use an existing chatbot?<span>+</span></summary><p>This project emphasizes biblical-language testing, source provenance, manuscript-aware retrieval, interpretation labels and public evaluation.</p></details>
          <details><summary>Is this project only for Seventh-day Adventists?<span>+</span></summary><p>No. Scripture is evaluated independently of denominational identity. Official church documents can verify what an organization teaches, but they do not replace biblical evidence or determine whether a doctrine is true.</p></details>
          <details><summary>Are contributions tax deductible?<span>+</span></summary><p>The project will not represent contributions as tax deductible unless a qualified fiscal sponsor or tax-exempt organization formally receives them. The approved campaign language will be posted before launch.</p></details>
        </div>
      </section>

      <section className="support" id="support">
        <div className="support-mark" aria-hidden="true">✦</div>
        <p className="section-label light">Be part of the foundation</p>
        <h2>Help build an AI that is<br /><em>answerable to its sources.</em></h2>
        <p>The public funding campaign is being prepared. Join the early-interest list, volunteer expertise, or discuss a founding sponsorship.</p>
        <div className="support-actions">
          <a className="button button-light" href="mailto:lonnie.lotus@gmail.com?subject=Bible-Grounded%20AI%20Project">Contact the project <span>→</span></a>
          <a className="outline-button" href="#roadmap">Review the budget</a>
        </div>
        <small>Support does not purchase control over doctrine, safety standards, datasets or research findings.</small>
      </section>

      <footer>
        <a className="brand footer-brand" href="#top"><span className="brand-mark" aria-hidden="true"><i /><i /><i /></span><span>Bible-Grounded <b>AI</b></span></a>
        <p>An independent, early-stage research initiative led by Lannys Lores.</p>
        <div><a href="#mission">Mission</a><a href="#roadmap">Budget</a><a href="#faq">FAQ</a></div>
        <small>© 2026 Bible-Grounded AI Initiative. Research concept—not spiritual, legal, medical or financial advice.</small>
      </footer>
    </main>
  );
}
