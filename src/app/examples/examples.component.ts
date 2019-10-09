import { Component, OnInit } from '@angular/core';
import { GithubService } from '../github.service';
import { Observable } from 'rxjs';
import { debounceTime, distinctUntilChanged, map } from 'rxjs/operators';


const showcaseList = ['NLP', 'benchmark', 'image_classification', 'image_detection', 'image_generation', 'image_segmentation', 'style_transfer', 'tabular'];

@Component({
  selector: 'app-examples',
  templateUrl: './examples.component.html',
  styleUrls: ['./examples.component.css'],
})
export class ExamplesComponent implements OnInit {

  examples: Array<any>;
  showcase: string;

  search = (text$: Observable<string>) =>
    text$.pipe(
      debounceTime(200),
      distinctUntilChanged(),
      map(term => term.length < 1 ? []
        : showcaseList.filter(v => v.toLowerCase().indexOf(term.toLowerCase()) > -1).slice(0, 10))
    )


  constructor(private githubService: GithubService) { }

  ngOnInit() {
    this.githubService.getExampleList()
      .subscribe((data) => {
        this.examples = (data as any[]).filter(e => (e.type == 'dir')).map(e => { return {name: e.name, url: e.html_url}; });
      });
  }

}
